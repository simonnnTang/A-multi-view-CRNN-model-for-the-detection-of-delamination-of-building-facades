import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, precision_score,recall_score
from tqdm import tqdm
import time
import sys
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ====================== 1. 全局配置 ======================
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FUSION_DIR = os.path.join(BASE_DIR, 'fusion')
os.makedirs(FUSION_DIR, exist_ok=True)
B2_INFO_PATH = os.path.join(BASE_DIR, "Branch Two", "preprocess_info.npz")

# 参数
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
EPOCHS = 50
LEARNING_RATE = 0.0008
WEIGHT_DECAY = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0
PIN_MEMORY = True if torch.cuda.is_available() else False
PATIENCE = 6
MIN_LR = 1e-7
EARLY_STOPPING_PATIENCE = 15
DROPOUT_RATE = 0.35

print("=" * 60)
print(f"Fusion 训练配置 (优化版):")
print(f"设备: {DEVICE}")
print(f"批次: {BATCH_SIZE} (累积{GRADIENT_ACCUMULATION_STEPS}步)")
print("=" * 60)


# ====================== 2. 数据集类 ======================
class FusionDataset(Dataset):
    def __init__(self, b1_data, b2_data):
        self.b1_data = b1_data
        self.b2_data = b2_data
        assert len(self.b1_data) == len(self.b2_data)

    def __len__(self):
        return len(self.b1_data)

    def __getitem__(self, idx):
        audio_tensor, label_b1 = self.b1_data[idx]
        spec_tensor, label_b2 = self.b2_data[idx]
        assert label_b1 == label_b2
        return audio_tensor, spec_tensor, label_b1


def prepare_data(spec_info, test_info, sr=22050):
    print("\n[步骤1/6] 准备融合数据...")
    if not os.path.exists(spec_info):
        raise FileNotFoundError(f"未找到: {spec_info}")

    data = np.load(spec_info, allow_pickle=True)
    audio_paths = data['audio_paths']
    spec_array = data['spec_array']
    le = LabelEncoder()
    labels = le.fit_transform(data['labels'])

    test_data = np.load(test_info, allow_pickle=True)
    test_audios = torch.tensor(test_data['audio_array'],dtype=torch.float32)
    test_specs = torch.tensor(test_data['spec_array'],dtype=torch.float32)
    test_labels = le.fit_transform(test_data['labels'])

    # 划分索引
    train_idx, val_idx = train_test_split(
        np.arange(len(labels)), test_size=0.2,
        random_state=SEED, stratify=labels
    )

    # 构建数据列表 (内存预加载)
    def collect(indices, name):
        b1_list, b2_list = [], []
        print(f"  正在加载 {name} 集 ({len(indices)}样本)...")
        for i in tqdm(indices):
            # Audio
            y, _ = librosa.load(audio_paths[i], sr=sr)
            b1_list.append((torch.tensor(y, dtype=torch.float32), labels[i]))
            # Spec
            b2_list.append((torch.tensor(spec_array[i], dtype=torch.float32), labels[i]))
        return b1_list, b2_list

    b1_train, b2_train = collect(train_idx, "Train")
    b1_val, b2_val = collect(val_idx, "Val")

    b1_test,b2_test=[],[]
    print(f"  正在加载测试集 ({len(test_labels)}样本)...")
    for i in range(len(test_labels)):
        b1_test.append((torch.tensor(test_audios[i]),test_labels[i]))
        b2_test.append((torch.tensor(test_specs[i]), test_labels[i]))

    # DataLoader
    print('\n[步骤2/6]创建dataloader...')
    train_dl = DataLoader(
        FusionDataset(b1_train, b2_train), batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True
    )
    val_dl = DataLoader(
        FusionDataset(b1_val, b2_val), batch_size=BATCH_SIZE * 2,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    test_dl = DataLoader(
        FusionDataset(b1_test, b2_test), batch_size=BATCH_SIZE * 2,
        shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    return train_dl, val_dl, test_dl, le


# ====================== 3. 模型定义 ======================
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=1)
        return torch.sum(x * w, dim=1)


class AudioBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 3, 1, 1), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(DROPOUT_RATE),
            nn.Conv1d(32, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(DROPOUT_RATE),
            nn.Conv1d(64, 128, 3, 1, 1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(DROPOUT_RATE)
        )
        self.lstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        self.attn = AttentionLayer(128)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        x = self.cnn(x).transpose(1, 2)
        x, _ = self.lstm(x)
        return self.attn(x)


class SpecBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(), nn.Dropout2d(DROPOUT_RATE / 2), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.Dropout2d(DROPOUT_RATE / 2), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout2d(DROPOUT_RATE / 2), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout2d(DROPOUT_RATE / 2),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((8, None))
        self.lstm = nn.LSTM(128, 64, num_layers=2, batch_first=True, bidirectional=True, dropout=DROPOUT_RATE)
        self.attn = AttentionLayer(128)

    def forward(self, x):
        x = (x - 0.5) / 0.5
        x = self.cnn(x)
        x = self.pool(x).permute(0, 3, 2, 1).contiguous()  # [B, W, 8, 128]
        B, W, H, C = x.size()
        x = x.view(B, W, -1)  # [B, W, 1024]
        # 注意：这里如果LSTM输入是1024，需要调整LSTM init参数，假设维持B2结构
        # 为匹配B2结构，这里做一个线性映射或调整LSTM
        # B2结构中 LSTM input=8*128=1024. 所以SpecBranch的LSTM input_size应为1024
        return self.attn(self.lstm(x)[0])


# 修正 SpecBranch LSTM 定义以匹配维度
class SpecBranchFixed(SpecBranch):
    def __init__(self):
        super().__init__()  # 继承基础结构
        # 覆盖 LSTM
        self.lstm = nn.LSTM(1024, 64, num_layers=2, batch_first=True, bidirectional=True, dropout=DROPOUT_RATE)


# 门控特征部分
class GatedFusion(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        # 1. 模态变换层 (将原始特征映射到隐空间)
        self.fc_audio = nn.Linear(dim, dim)
        self.fc_spec = nn.Linear(dim, dim)

        # 2. 门控生成层 (Gate Generation Network)
        # 输入是拼接后的特征 (2*dim)，输出是一个 0-1 之间的权重向量 (dim)
        self.gate_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()  # 关键：Sigmoid 将输出限制在 [0, 1] 之间作为权重
        )

    def forward(self, audio_feat, spec_feat):
        # audio_feat: [B, 128], spec_feat: [B, 128]

        # A. 特征变换 (Feature Transformation)
        h_audio = torch.tanh(self.fc_audio(audio_feat))
        h_spec = torch.tanh(self.fc_spec(spec_feat))

        # B. 计算门控权重 (Gate Computation)
        # 将两个特征拼接，让网络“看”到整体情况，然后决定权重
        combined = torch.cat([audio_feat, spec_feat], dim=1)
        z = self.gate_net(combined)  # z 是 audio 的“信任度”

        # C. 加权融合 (Weighted Fusion)
        # 公式: H = z * H_audio + (1 - z) * H_spec
        # 如果 z 接近 1，模型主要看 Audio；如果 z 接近 0，主要看 Spectrum
        fused_feat = z * h_audio + (1 - z) * h_spec

        return fused_feat


class JointFusionModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.audio = AudioBranch()
        self.spec = SpecBranchFixed()
        self.fusion = GatedFusion(dim=128)
        self.classifier = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(64,num_classes),
        )
        self._init_weights()
        self.to(DEVICE)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.Linear)):
                if hasattr(m, 'weight') and m.weight is not None: nn.init.normal_(m.weight, 0, 0.01)
                if hasattr(m, 'bias') and m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, a, s):
        fa = self.audio(a)
        fs = self.spec(s)
        fused=self.fusion(fa,fs)
        return self.classifier(fused)


# ====================== 4. 训练核心 ======================
def train_model(model, train_loader, val_loader, criterion, optimizer):
    print("\n[步骤4/6] 开始训练 (优化版)...")
    scaler = GradScaler() if torch.cuda.is_available() else None
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=PATIENCE, min_lr=MIN_LR
    )

    best_f1 = 0.0
    early_cnt = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    save_path = os.path.join(FUSION_DIR, 'best_fusion_model.pth')

    start = time.time()

    for epoch in range(EPOCHS):
        model.train()
        t_loss = 0.0
        t_corr = 0
        t_total = 0
        optimizer.zero_grad()

        loop = tqdm(train_loader, desc=f"Ep {epoch + 1}/{EPOCHS}")
        for i, (a, s, y) in enumerate(loop):
            a, s, y = a.to(DEVICE), s.to(DEVICE), y.to(DEVICE)

            with autocast(enabled=scaler is not None):
                out = model(a, s)
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

            t_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS * a.size(0)
            _, p = torch.max(out, 1)
            t_corr += (p == y).sum().item()
            t_total += y.size(0)
            loop.set_postfix(loss=loss.item() * GRADIENT_ACCUMULATION_STEPS)

        # Val
        model.eval()
        v_loss = 0.0
        v_preds, v_true = [], []
        with torch.no_grad():
            for a, s, y in val_loader:
                a, s, y = a.to(DEVICE), s.to(DEVICE), y.to(DEVICE)
                out = model(a, s)
                loss = criterion(out, y)
                v_loss += loss.item() * a.size(0)
                _, p = torch.max(out, 1)
                v_preds.extend(p.cpu().numpy())
                v_true.extend(y.cpu().numpy())

        # Metrics
        e_t_loss = t_loss / t_total
        e_t_acc = t_corr / t_total
        e_v_loss = v_loss / len(val_loader.dataset)
        e_v_acc = accuracy_score(v_true, v_preds)
        e_v_f1 = f1_score(v_true, v_preds, average='weighted')

        history['train_loss'].append(e_t_loss)
        history['val_loss'].append(e_v_loss)
        history['train_acc'].append(e_t_acc)
        history['val_acc'].append(e_v_acc)

        scheduler.step(e_v_f1)
        print(f"  Train -> Loss: {e_t_loss:.4f} | Acc: {e_t_acc:.4f}")
        print(f"  Val   -> Loss: {e_v_loss:.4f} | Acc: {e_v_acc:.4f} | F1: {e_v_f1:.4f}")

        if e_v_f1 > best_f1:
            best_f1 = e_v_f1
            early_cnt = 0
            torch.save(model.state_dict(), save_path)
            print(f"  ✅ Saved Best Model (F1: {best_f1:.4f})")
        else:
            early_cnt += 1
            print(f"  ⏳ Early Stopping: {early_cnt}/{EARLY_STOPPING_PATIENCE}")

        # if early_cnt >= EARLY_STOPPING_PATIENCE: break

    print(f"Total Time: {(time.time() - start) / 60:.2f} min")
    return history, save_path


# ====================== 5. 评估与绘图 ======================
def plot_results(history):
    print('\n[步骤5/6]绘制曲线...')
    # 创建2个子图，上下排列
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=150, sharex=True)
    fig.suptitle('Fusion', fontsize=20, fontweight='bold')

    # 绘制损失曲线
    epochs = range(len(history['train_loss']))
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2, marker='o', markersize=4)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2, marker='s', markersize=4)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Convergence Curve', fontsize=14)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_ylim(bottom=0)

    # 绘制准确率曲线
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2, marker='o', markersize=4)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Accuracy Convergence Curve', fontsize=14)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.set_ylim(0, 1.05)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    curve_save_path = os.path.join(FUSION_DIR, "training_curves.png")
    plt.savefig(curve_save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def evaluate(model, loader, class_names):
    print('\n[步骤6/6]最终评价...')
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for a, s, y in loader:
            a, s, y = a.to(DEVICE), s.to(DEVICE), y.to(DEVICE)
            with torch.amp.autocast(device_type='cuda'):
                out = model(a, s)
                _, p = torch.max(out, 1)
                preds.extend(p.cpu().numpy())
                targets.extend(y.cpu().numpy())

    accuracy = accuracy_score(targets,preds)
    precision = precision_score(targets, preds, average='weighted')
    recall = recall_score(targets, preds, average='weighted')
    f1 = f1_score(targets, preds, average='weighted')
    report = classification_report(targets, preds, target_names=class_names,digits=4)
    cm = confusion_matrix(targets, preds)

    print(report)
    with open(os.path.join(FUSION_DIR, 'fusion_metrics.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Accuracy: {accuracy:.4f}\nF1 Score: {f1:.4f}\nPrecision:{precision:.4f}"
                f"Recall:{recall:.4f}\n\n{report}")

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predict', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.savefig(os.path.join(FUSION_DIR, 'fusion_cm.png'))
    plt.close()


# ====================== 主程序 ======================
if __name__ == '__main__':
    print(f"程序启动时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    test_path = os.path.join(BASE_DIR,'test_info.npz')

    # 1. 数据
    train_dl, val_dl, test_dl, le = prepare_data(B2_INFO_PATH, test_path)

    # 2. 模型
    print('\n[步骤3/6]初始化模型...')
    model = JointFusionModel(num_classes=2)

    # 3. 优化
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, amsgrad=True)

    # 4. 运行
    hist, path = train_model(model, train_dl, val_dl, criterion, optimizer)

    # 5. 结果
    model.load_state_dict(torch.load(path))
    plot_results(hist)
    evaluate(model, test_dl, le.classes_)
    print("\nFusion 训练结束！")
    print(f"程序结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")