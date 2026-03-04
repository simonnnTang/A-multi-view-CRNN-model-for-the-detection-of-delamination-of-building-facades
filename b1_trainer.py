import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pyexpat import features

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
import sys
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm

# 关闭警告
warnings.filterwarnings('ignore')

# ====================== 1. 全局配置 ======================
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BRANCH_ONE_DIR = os.path.join(BASE_DIR, 'Branch One')
os.makedirs(BRANCH_ONE_DIR, exist_ok=True)

# 训练参数
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
EPOCHS = 50
LEARNING_RATE = 0.0008
WEIGHT_DECAY = 2e-5
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0  # Windows 建议 0
PIN_MEMORY = True if torch.cuda.is_available() else False
PATIENCE = 6
MIN_LR = 1e-7
EARLY_STOPPING_PATIENCE = 15
DROPOUT_RATE = 0.35

print("=" * 60)
print(f"B1 训练配置 (优化版):")
print(f"设备: {DEVICE}")
print(f"批次: {BATCH_SIZE} (累积{GRADIENT_ACCUMULATION_STEPS}步 -> 等效{BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
print(f"学习率: {LEARNING_RATE}")
print("=" * 60)


# ====================== 2. 数据处理 ======================
class AudioDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label


def load_data_and_encoder(data_path, encoder_path):
    print("\n[步骤1/6] 加载数据...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"未找到数据文件: {data_path}")

    data = np.load(data_path)
    X_train, X_val = data['X_train'], data['X_val']
    y_train, y_val = data['y_train'], data['y_val']

    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)

    print(f"  训练集: {len(X_train)} | 验证集: {len(X_val)}")
    return X_train, y_train, X_val, y_val, label_encoder


def create_dataloaders(X_train, y_train, X_val, y_val):
    print('\n[2/6]创建dataloader...')
    train_dataset = AudioDataset(X_train, y_train)
    val_dataset = AudioDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE*2, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    return train_loader, val_loader


# ====================== 3. 模型定义 (增强稳定性) ======================
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.u = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        e = torch.tanh(self.W(x))
        e = self.u(e).squeeze(-1)
        attention_weights = torch.softmax(e, dim=1).unsqueeze(-1)
        output = x * attention_weights
        return torch.sum(output, dim=1)


class AudioModel(nn.Module):
    def __init__(self, input_dim=1, num_classes=2):
        super(AudioModel, self).__init__()

        # CNN部分
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(DROPOUT_RATE),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(DROPOUT_RATE),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(DROPOUT_RATE)
        )

        # LSTM
        self.lstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True, num_layers=1)
        self.dropout_lstm = nn.Dropout(DROPOUT_RATE)

        # Attention
        self.attention = AttentionLayer(128)

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(64, num_classes)
        )

        self._init_weights()
        self.to(DEVICE)

    def _init_weights(self):
        """Kaiming 初始化，提升收敛速度"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.Linear)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.normal_(m.weight, 0, 0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        x = self.cnn(x)  # [B, 128, L/8]
        x = x.transpose(1, 2)  # [B, L/8, 128]
        x, _ = self.lstm(x)
        x = self.dropout_lstm(x)
        x = self.attention(x)
        x = self.fc(x)
        return x


# ====================== 4. 训练流程 (复制B2核心) ======================
def train_model(model, train_loader, val_loader, criterion, optimizer):
    print("\n[步骤4/6] 开始训练 (优化版)...")
    scaler = GradScaler() if torch.cuda.is_available() else None
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=PATIENCE, min_lr=MIN_LR
    )

    best_val_f1 = 0.0
    best_val_acc = 0.0
    early_stop_cnt = 0
    save_path = os.path.join(BRANCH_ONE_DIR, 'best_audio_model.pth')

    metrics = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }

    start_time = time.time()

    for epoch in range(EPOCHS):
        # --- 训练 ---
        model.train()
        train_loss = 0.0
        train_preds, train_true = [], []
        optimizer.zero_grad()

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

            with autocast(enabled=scaler is not None):
                out = model(x)
                loss = criterion(out, y) / GRADIENT_ACCUMULATION_STEPS

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * x.size(0) * GRADIENT_ACCUMULATION_STEPS
            _, pred = torch.max(out, 1)
            train_preds.extend(pred.cpu().numpy())
            train_true.extend(y.cpu().numpy())

        # --- 验证 ---
        model.eval()
        val_loss = 0.0
        val_preds, val_true = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                with autocast(enabled=scaler is not None):
                    out = model(x)
                    loss = criterion(out, y)
                    val_loss += loss.item() * x.size(0)
                    _, pred = torch.max(out, 1)
                    val_preds.extend(pred.cpu().numpy())
                    val_true.extend(y.cpu().numpy())

        # --- 指标计算 ---
        t_loss = train_loss / len(train_loader.dataset)
        v_loss = val_loss / len(val_loader.dataset)
        t_acc = accuracy_score(train_true, train_preds)
        v_acc = accuracy_score(val_true, val_preds)
        v_f1 = f1_score(val_true, val_preds, average='weighted', zero_division=0)

        metrics['train_loss'].append(t_loss)
        metrics['val_loss'].append(v_loss)
        metrics['train_acc'].append(t_acc)
        metrics['val_acc'].append(v_acc)

        # --- 学习率 & 早停 ---
        scheduler.step(v_f1)
        curr_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch + 1}/{EPOCHS} | LR: {curr_lr:.2e}")
        print(f"  Train -> Loss: {t_loss:.4f} | Acc: {t_acc:.4f}")
        print(f"  Val   -> Loss: {v_loss:.4f} | Acc: {v_acc:.4f} | F1: {v_f1:.4f}")

        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            best_val_acc = v_acc
            early_stop_cnt = 0
            torch.save(model.state_dict(), save_path)
            print(f"  ✅ 最佳模型已保存 (F1: {best_val_f1:.4f})")
        else:
            early_stop_cnt += 1
            print(f"  ⏳ 早停计数: {early_stop_cnt}/{EARLY_STOPPING_PATIENCE}")

        # if early_stop_cnt >= EARLY_STOPPING_PATIENCE:
        #     print("🛑 触发早停")
        #     break

    print(f"\n训练结束，耗时 {(time.time() - start_time) / 60:.2f} 分钟")
    return metrics, save_path


# ====================== 5. 结果可视化 ======================
def plot_curves(metrics):
    print("\n[步骤5/6] 绘制曲线...")
    # 创建2个子图，上下排列
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=150, sharex=True)
    fig.suptitle('Branch 1', fontsize=20, fontweight='bold')

    # 绘制损失曲线
    epochs = range(len(metrics['train_loss']))
    ax1.plot(epochs, metrics['train_loss'], 'b-', label='Train', linewidth=2, marker='o', markersize=4)
    ax1.plot(epochs, metrics['val_loss'], 'r-', label='Val', linewidth=2, marker='s', markersize=4)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Convergence Curve', fontsize=14)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_ylim(bottom=0)

    # 绘制准确率曲线
    ax2.plot(epochs, metrics['train_acc'], 'b-', label='Train', linewidth=2, marker='o', markersize=4)
    ax2.plot(epochs, metrics['val_acc'], 'r-', label='Val', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Accuracy Convergence Curve', fontsize=14)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.set_ylim(0, 1.05)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    curve_save_path = os.path.join(BRANCH_ONE_DIR, "training_curves.png")
    plt.savefig(curve_save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def evaluate_final(model, val_loader, class_names):
    print("\n[步骤6/6] 最终评估...")
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE, non_blocking=True, dtype=torch.float32)
            y = y.to(DEVICE, non_blocking=True, dtype=torch.float32)
            with torch.amp.autocast(device_type='cuda'):
                out = model(x)
                _, p = torch.max(out.data, 1)
                preds.extend(p.cpu().numpy())
                targets.extend(y.cpu().numpy())

    # 指标
    accuracy = accuracy_score(targets,preds)
    precision = precision_score(targets, preds, average='weighted')
    recall = recall_score(targets, preds, average='weighted')
    f1 = f1_score(targets, preds, average='weighted')
    report = classification_report(targets, preds, target_names=class_names, digits=4)
    cm = confusion_matrix(targets, preds)

    print(report)

    # 保存指标
    with open(os.path.join(BRANCH_ONE_DIR, 'b1_evaluation_metrics.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Accuracy: {accuracy:.4f}\nF1 Score: {f1:.4f}\nPrecision:{precision:.4f}"
                f"Recall:{recall:.4f}\n\n{report}")


    # 混淆矩阵
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predict', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(BRANCH_ONE_DIR, 'confusion_matrix.png'))
    plt.close()


# ====================== 主程序 ======================
if __name__ == '__main__':
    print(f"程序启动时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    data_path = os.path.join(BRANCH_ONE_DIR, 'processed_data.npz')
    encoder_path = os.path.join(BRANCH_ONE_DIR, 'label_encoder.pkl')

    test_path = os.path.join(BASE_DIR,'test_data.npz')

    # 1. 加载
    # 训练验证集
    X_train, y_train, X_val, y_val, le = load_data_and_encoder(data_path, encoder_path)
    train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val)

    # 测试集
    test_data = np.load(test_path)
    X_test, y_test = test_data['features'], test_data['labels']
    y_test_encode = le.fit_transform(y_test)
    test_dataset = AudioDataset(X_test, y_test_encode)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE*2, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True
    )

    # 2. 模型
    print("\n[步骤3/6] 初始化模型和优化器（泛化优化版）...")
    model = AudioModel(num_classes=len(le.classes_))

    # 3. 优化器 (AMSGrad + LabelSmoothing)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15).to(DEVICE)
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, amsgrad=True
    )

    # 4. 训练
    history, best_path = train_model(model, train_loader, val_loader, criterion, optimizer)

    # 5. 评估
    model.load_state_dict(torch.load(best_path))
    plot_curves(history)
    evaluate_final(model, test_loader, le.classes_)

    print("\nB1 训练流程结束！")
    print(f"程序结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")