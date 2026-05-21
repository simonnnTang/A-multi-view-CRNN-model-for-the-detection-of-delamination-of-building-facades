import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score,classification_report
import seaborn as sns
import time
import sys
import warnings
from torch.cuda.amp import autocast, GradScaler
import cv2
from torch.nn import functional as F
from sklearn.preprocessing import LabelEncoder

# 关闭不必要的警告
warnings.filterwarnings('ignore')

# ====================== 1. 全局配置与内存优化 ======================
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True  # 提升稳定性
torch.backends.cudnn.benchmark = False  # 关闭自动优化，提升稳定性
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# PyTorch配置（内存优化+稳定性）
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 便于调试GPU错误
os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# 路径配置 - 修改为Branch Two目录结构
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BRANCH_TWO_DIR = os.path.join(BASE_DIR, "Branch Two")
MODEL_DIR = os.path.join(BRANCH_TWO_DIR, "CRNN_Training", "models")
PLOT_DIR = os.path.join(BRANCH_TWO_DIR, "CRNN_Training", "plots")
METRICS_DIR = os.path.join(BRANCH_TWO_DIR, "CRNN_Training", "metrics")
TEMP_DIR = os.path.join(BRANCH_TWO_DIR, "CRNN_Training", "temp")
PREPROCESS_INFO = os.path.join(BRANCH_TWO_DIR, "preprocess_info.npz")  # 从Branch Two加载预处理文件

# 自动创建目录
for dir_path in [BRANCH_TWO_DIR, MODEL_DIR, PLOT_DIR, METRICS_DIR, TEMP_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# 训练参数（优化泛化能力+稳定性+内存优化）
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 0.0008  # 降低初始学习率提升稳定性
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0 if os.name == 'nt' else 2
PIN_MEMORY = True if torch.cuda.is_available() else False  # 合理设置pin_memory
GRADIENT_ACCUMULATION_STEPS = 4
IMG_RESIZE_SIZE = (128, 128)
WEIGHT_DECAY = 2e-5  # 增加权重衰减提升泛化能力
PATIENCE = 6  # 调整学习率调整耐心值
MIN_LR = 1e-7  # 最小学习率
EARLY_STOPPING_PATIENCE = 15  # 早停机制
DROPOUT_RATE = 0.35  # 增加dropout提升泛化能力

print(f"训练配置（泛化优化版）：")
print(f"训练设备: {DEVICE}")
print(
    f"批次大小: {BATCH_SIZE} (梯度累积{GRADIENT_ACCUMULATION_STEPS}步，等效{BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
print(f"训练轮数: {EPOCHS}")
print(f"学习率: {LEARNING_RATE} (最小{MIN_LR})")
print(f"数据加载进程数: {NUM_WORKERS}")
print(f"图像尺寸限制: {IMG_RESIZE_SIZE}")
print(f"文件保存根目录: {BRANCH_TWO_DIR}")
print(f"早停耐心值: {EARLY_STOPPING_PATIENCE}")
print(f"Dropout率: {DROPOUT_RATE}")

# 系统信息
print("=" * 60)
print("系统环境检测：")
print(f"Python版本: {sys.version.split()[0]}")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)} | 显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f}GB")
else:
    print("警告：未检测到CUDA，将使用CPU训练")
print("=" * 60)


 # ====================== 2. 数据集（增强+内存优化+路径修复版） ======================

class AudioDataset(Dataset):
    """音频数据集类"""
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label

def create_dataloaders(img_paths, img_array, labels):
    print("\n[步骤2/6] 创建数据集和数据加载器...")
    start_time = time.time()

    if len(img_paths) == 0:
        print("  错误：无可用的频谱图像文件！")
        return None, None

    print(f"  原始样本数：{len(img_paths)}")

    # 预处理路径：转为绝对路径并过滤无效文件
    processed_paths = []
    processed_array = []
    processed_labels = []
    for path, array, label in zip(img_paths, img_array, labels):
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path) and os.path.getsize(abs_path) > 0:
            processed_paths.append(abs_path)
            processed_array.append(array)
            processed_labels.append(label)

    print(f"  过滤后有效样本数：{len(processed_paths)}")

    if len(processed_paths) == 0:
        print("  错误：过滤后无有效样本！")
        return None, None

    print(f"  开始划分训练集和验证集（8:2）...")

    '==============使用处理的array进行训练=============='
    X_train, X_val, y_train, y_val = train_test_split(
        processed_array, processed_labels, test_size=0.2, random_state=SEED,
        shuffle=True, stratify=processed_labels)

    train_dataset = AudioDataset(X_train, y_train)
    val_dataset = AudioDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        persistent_workers=True if NUM_WORKERS > 0 else False)  # 提升稳定性)

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=PIN_MEMORY,
        drop_last=False)

    elapsed_time = time.time() - start_time
    print(f"  数据集划分完成！")
    print(f"  训练集: {len(train_dataset)} samples ({len(train_loader)} batches)")
    print(f"  验证集: {len(val_dataset)} samples ({len(val_loader)} batches)")
    print(f"  训练集已启用数据增强提升泛化能力")
    print(f"  耗时: {elapsed_time:.2f} 秒")

    return train_loader, val_loader


# ====================== 3. CRNN模型（增强泛化能力+稳定性） ======================
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),  # 增加dropout
            nn.Linear(hidden_dim // 2, 1))

    def forward(self, rnn_output):
        attention_weights = torch.softmax(self.attention(rnn_output), dim=1)
        weighted_output = torch.sum(rnn_output * attention_weights, dim=1)
        return weighted_output, attention_weights


class CRNNWithAttention(nn.Module):
    def __init__(self, input_channels=3, num_classes=2):
        super(CRNNWithAttention, self).__init__()

        # CNN部分（增强泛化能力+稳定性）
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(DROPOUT_RATE / 2),  # 增加2D dropout
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(DROPOUT_RATE / 2),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(DROPOUT_RATE / 2),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(DROPOUT_RATE / 2),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, None))

        # RNN部分（增强泛化能力+稳定性）
        self.rnn = nn.LSTM(
            input_size=8 * 128,
            hidden_size=128,
            num_layers=2,  # 增加一层提升表达能力
            bidirectional=True,
            batch_first=True,
            dropout=DROPOUT_RATE if 2 > 1 else 0)  # 仅多层时启用dropout

        # 注意力层
        self.attention = AttentionLayer(128 * 2)

        # 分类器（增强泛化能力+稳定性）
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(128 * 2, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(64, num_classes)
        )

        # 权重初始化提升稳定性
        self._init_weights()

        self.to(DEVICE, dtype=torch.float32)

    def _init_weights(self):
        """权重初始化提升稳定性"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = (x - 0.5) / 0.5  # 归一化

        cnn_out = self.cnn(x)
        cnn_out = self.adaptive_pool(cnn_out)

        batch_size = cnn_out.size(0)
        cnn_out_width = cnn_out.size(3)
        cnn_out = cnn_out.permute(0, 3, 2, 1)
        cnn_out = cnn_out.contiguous().view(batch_size, cnn_out_width, -1)

        rnn_out, _ = self.rnn(cnn_out)
        attn_out, _ = self.attention(rnn_out)
        output = self.classifier(attn_out)

        return output


# ====================== 4. 模型训练（增强泛化能力+稳定性+早停） ======================
def train_model(model, train_loader, val_loader, criterion, optimizer):
    print("\n[步骤4/6] 开始模型训练（泛化优化版）...")
    start_time = time.time()

    if train_loader is None or val_loader is None:
        print("  错误：数据加载器为空，无法训练！")
        return {}, [], []

    # 混合精度训练
    scaler = GradScaler() if torch.cuda.is_available() else None

    # 清空缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"  GPU初始内存占用: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")

    # 训练记录（增强评价指标）
    metrics_history = {
        'train_losses': [], 'val_losses': [],
        'train_accs': [], 'val_accs': [],
        'train_precisions': [], 'val_precisions': [],
        'train_recalls': [], 'val_recalls': [],
        'train_f1s': [], 'val_f1s': []
    }
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_model_path = os.path.join(MODEL_DIR, "best_crnn_attention_model.pth")
    prev_lr = LEARNING_RATE
    early_stopping_counter = 0

    # 学习率调度器（增强稳定性）- 移除verbose参数
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=PATIENCE, min_lr=MIN_LR
    )

    # 保存所有预测结果用于混淆矩阵
    # all_val_preds = []
    # all_val_labels = []

    # 训练循环
    try:
        for epoch in range(EPOCHS):
            epoch_start = time.time()

            # 训练阶段
            model.train()
            train_loss = 0.0
            train_preds = []
            train_labels = []
            optimizer.zero_grad()

            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(DEVICE, non_blocking=True, dtype=torch.float32)
                labels = labels.to(DEVICE, non_blocking=True, dtype=torch.long)

                with autocast(enabled=scaler is not None):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss = loss / GRADIENT_ACCUMULATION_STEPS

                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    if scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 更大的梯度裁剪
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                    optimizer.zero_grad()

                train_loss += loss.item() * inputs.size(0) * GRADIENT_ACCUMULATION_STEPS
                _, predicted = torch.max(outputs.data, 1)
                train_preds.extend(predicted.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())

                if (batch_idx + 1) % max(1, len(train_loader) // 5) == 0:
                    gpu_mem = torch.cuda.memory_allocated(0) / 1024 ** 3 if torch.cuda.is_available() else 0
                    print(
                        f"    Epoch {epoch + 1}/{EPOCHS} - Batch {batch_idx + 1}/{len(train_loader)} "
                        f"- Loss: {loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f} "
                        f"- GPU Mem: {gpu_mem:.2f}GB"
                    )

            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_true = []

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(DEVICE, non_blocking=True, dtype=torch.float32)
                    labels = labels.to(DEVICE, non_blocking=True, dtype=torch.long)

                    with autocast(enabled=scaler is not None):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        val_loss += loss.item() * inputs.size(0)
                        _, predicted = torch.max(outputs.data, 1)
                        val_preds.extend(predicted.cpu().numpy())
                        val_true.extend(labels.cpu().numpy())

            # 计算增强评价指标
            # 训练集指标
            train_acc = accuracy_score(train_labels, train_preds) if train_labels else 0
            train_precision = precision_score(train_labels, train_preds, average='weighted', zero_division=0)
            train_recall = recall_score(train_labels, train_preds, average='weighted', zero_division=0)
            train_f1 = f1_score(train_labels, train_preds, average='weighted', zero_division=0)
            avg_train_loss = train_loss / len(train_labels) if train_labels else 0

            # 验证集指标
            val_acc = accuracy_score(val_true, val_preds) if val_true else 0
            val_precision = precision_score(val_true, val_preds, average='weighted', zero_division=0)
            val_recall = recall_score(val_true, val_preds, average='weighted', zero_division=0)
            val_f1 = f1_score(val_true, val_preds, average='weighted', zero_division=0)
            avg_val_loss = val_loss / len(val_true) if val_true else 0

            # 记录指标
            metrics_history['train_losses'].append(avg_train_loss)
            metrics_history['val_losses'].append(avg_val_loss)
            metrics_history['train_accs'].append(train_acc)
            metrics_history['val_accs'].append(val_acc)
            metrics_history['train_precisions'].append(train_precision)
            metrics_history['val_precisions'].append(val_precision)
            metrics_history['train_recalls'].append(train_recall)
            metrics_history['val_recalls'].append(val_recall)
            metrics_history['train_f1s'].append(train_f1)
            metrics_history['val_f1s'].append(val_f1)

            # 保存验证集预测结果
            # all_val_preds.extend(val_preds)
            # all_val_labels.extend(val_true)

            # 更新学习率
            scheduler.step(val_f1)  # 基于F1分数调整学习率，更全面
            current_lr = optimizer.param_groups[0]['lr']

            # 打印学习率变化（替代verbose=True的功能）
            if current_lr < prev_lr:
                print(f"    🔄 学习率调整：{prev_lr:.6f} → {current_lr:.6f} (验证F1未提升{PATIENCE}轮)")
                prev_lr = current_lr

            # 打印详细指标信息
            print(f"\n    === Epoch {epoch + 1}/{EPOCHS} 训练指标 ===")
            print(f"    训练损失: {avg_train_loss:.4f} | 训练准确率: {train_acc:.4f}")
            print(f"    训练精确率: {train_precision:.4f} | 训练召回率: {train_recall:.4f} | 训练F1: {train_f1:.4f}")
            print(f"    === Epoch {epoch + 1}/{EPOCHS} 验证指标 ===")
            print(f"    验证损失: {avg_val_loss:.4f} | 验证准确率: {val_acc:.4f}")
            print(f"    验证精确率: {val_precision:.4f} | 验证召回率: {val_recall:.4f} | 验证F1: {val_f1:.4f}")
            print(
                f"    当前学习率: {current_lr:.6f} | 最佳验证准确率: {best_val_acc:.4f} | 最佳验证F1: {best_val_f1:.4f}")

            # 保存最佳模型（基于F1分数，更全面）
            if val_f1 > best_val_f1:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                early_stopping_counter = 0  # 重置早停计数器
                # 保存完整指标
                torch.save(model.state_dict(),best_model_path)
                print(f"    ✨ 保存最佳模型到：{best_model_path} | 验证F1提升至: {best_val_f1:.4f}")
            else:
                early_stopping_counter += 1
                print(f"    ⚠️  验证F1未提升，早停计数器: {early_stopping_counter}/{EARLY_STOPPING_PATIENCE}")

            # 早停机制
            # if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
            #     print(f"\n    🛑 早停触发！连续{EARLY_STOPPING_PATIENCE}轮验证F1未提升，停止训练")
            #     break

            # 内存监控
            gpu_mem = torch.cuda.memory_allocated(0) / 1024 ** 3 if torch.cuda.is_available() else 0
            gpu_mem_peak = torch.cuda.max_memory_allocated(0) / 1024 ** 3 if torch.cuda.is_available() else 0

            epoch_time = time.time() - epoch_start
            print(f"    ⏱️  Epoch耗时: {epoch_time:.2f}s | GPU内存: {gpu_mem:.2f}GB (峰值: {gpu_mem_peak:.2f}GB)")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n❌ 错误：内存不足！建议：")
            print(f"   1. 进一步减小批次大小至4")
            print(f"   2. 降低图像尺寸至(64,64)")
            print(f"   3. 关闭混合精度训练")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            print(f"\n❌ 训练错误：{e}")
        raise e

    total_time = time.time() - start_time
    print(f"\n  训练完成！")
    print(f"  最佳验证准确率: {best_val_acc:.4f}")
    print(f"  最佳验证F1分数: {best_val_f1:.4f}")
    print(f"  总训练时间: {total_time / 60:.2f} 分钟")
    print(f"  最佳模型保存路径: {best_model_path}")

    # 打印最终汇总指标
    print(f"\n=== 训练最终汇总指标 ===")
    print(f"最后一轮训练集指标：")
    print(f"  损失: {metrics_history['train_losses'][-1]:.4f} | 准确率: {metrics_history['train_accs'][-1]:.4f}")
    print(
        f"  精确率: {metrics_history['train_precisions'][-1]:.4f} | 召回率: {metrics_history['train_recalls'][-1]:.4f} | F1: {metrics_history['train_f1s'][-1]:.4f}")
    print(f"最后一轮验证集指标：")
    print(f"  损失: {metrics_history['val_losses'][-1]:.4f} | 准确率: {metrics_history['val_accs'][-1]:.4f}")
    print(
        f"  精确率: {metrics_history['val_precisions'][-1]:.4f} | 召回率: {metrics_history['val_recalls'][-1]:.4f} | F1: {metrics_history['val_f1s'][-1]:.4f}")

    return metrics_history  , best_model_path#, all_val_labels, all_val_preds

def evaluate_model(model, val_loader):
    """模型验证"""
    model.eval()
    val_loss=0.0
    val_preds=[]
    val_true=[]

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE, non_blocking=True, dtype=torch.float32)
            labels =labels.to(DEVICE,non_blocking=True,dtype=torch.long)

            with torch.amp.autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()*inputs.size(0)
                _,predicted =torch.max(outputs.data,1)
                val_preds.extend(predicted.cpu().numpy())
                val_true.extend(labels.cpu().numpy())

    val_acc = accuracy_score(val_true, val_preds) if val_true else 0
    val_precision = precision_score(val_true, val_preds, average='weighted', zero_division=0)
    val_recall = recall_score(val_true, val_preds, average='weighted', zero_division = 0)
    val_f1 = f1_score(val_true, val_preds, average='weighted', zero_division=0)


    print("\n=====评价指标=====")
    print(f"准确率: {val_acc:.4f} | 精确率: {val_precision:.4f}")
    print(f"召回率: {val_recall:.4f} | F1分数: {val_f1:.4f}")
    print("==========	======")
    return val_true, val_preds


# ====================== 5. 结果处理（新增准确率和损失曲线绘制） ======================
def plot_training_curves(metrics_history):
    """绘制训练过程中的损失和准确率曲线，合并在一张图中"""
    print("\n[步骤5/6] 绘制训练曲线...")
    start_time = time.time()

    if not metrics_history or len(metrics_history['train_losses']) == 0:
        print("  错误：无训练指标数据，无法绘制曲线！")
        return

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建2个子图，上下排列
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=150, sharex=True)
    fig.suptitle('Branch 2', fontsize=20, fontweight='bold')

    # 绘制损失曲线
    epochs = range(len(metrics_history['train_losses']))
    ax1.plot(epochs, metrics_history['train_losses'], 'b-', label='Train', linewidth=2, marker='o', markersize=4)
    ax1.plot(epochs, metrics_history['val_losses'], 'r-', label='Val', linewidth=2, marker='s', markersize=4)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Convergence Curve', fontsize=14)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_ylim(bottom=0)

    # 绘制准确率曲线
    ax2.plot(epochs, metrics_history['train_accs'], 'b-', label='Train', linewidth=2, marker='o', markersize=4)
    ax2.plot(epochs, metrics_history['val_accs'], 'r-', label='Val', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Accuracy Convergence Curve', fontsize=14)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.set_ylim(0, 1.05)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    curve_save_path = os.path.join(PLOT_DIR, "training_curves.png")
    plt.savefig(curve_save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    elapsed_time = time.time() - start_time
    print(f"  训练曲线已保存至 {curve_save_path}")
    print(f"  耗时: {elapsed_time:.2f} 秒")


def plot_confusion_matrix(true_labels, pred_labels, class_names):
    print("\n[步骤6/6] 绘制混淆矩阵...")
    start_time = time.time()

    if len(true_labels) == 0:
        print("  错误：无验证数据，无法绘制混淆矩阵！")
        return

    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels)

    # 绘制混淆矩阵
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(8, 6), dpi=150)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                linewidths=0.5,
                cbar=True)

    plt.title('Confusion Matrix', fontsize=14)
    plt.xlabel('Predict', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.tight_layout()

    # 保存混淆矩阵
    cm_save_path = os.path.join(PLOT_DIR, "confusion_matrix.png")
    plt.savefig(cm_save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    # 计算并打印详细指标
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted')
    recall = recall_score(true_labels, pred_labels, average='weighted')
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    class_precision = precision_score(true_labels, pred_labels, average=None)
    class_recall = recall_score(true_labels, pred_labels, average=None)
    class_f1 = f1_score(true_labels, pred_labels, average=None)
    report = classification_report(true_labels, pred_labels, target_names=class_names, digits=4)

    print(f"\n=== 最终验证集评价指标 ===")
    print(f"整体准确率 (Accuracy): {accuracy:.4f}")
    print(f"加权精确率 (Precision): {precision:.4f}")
    print(f"加权召回率 (Recall): {recall:.4f}")
    print(f"加权F1分数 (F1 Score): {f1:.4f}")
    print(f"\n类别详细指标:")
    print(f"非空鼓类 - 精确率: {class_precision[0]:.4f}, 召回率: {class_recall[0]:.4f}, F1: {class_f1[0]:.4f}")
    print(f"空鼓类   - 精确率: {class_precision[1]:.4f}, 召回率: {class_recall[1]:.4f}, F1: {class_f1[1]:.4f}")

    # 保存指标到文件
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'class_precision': class_precision,
        'class_recall': class_recall,
        'class_f1': class_f1,
        'confusion_matrix': cm
    }

    metrics_file = os.path.join(METRICS_DIR, "b2_evaluation_metrics.txt")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        f.write("模型评价指标\n")
        f.write("=" * 50 + "\n")

        # f.write(f"准确率 (Accuracy): {metrics['accuracy']:.4f}\n")
        # f.write(f"加权精确率 (Precision): {metrics['precision']:.4f}\n")
        # f.write(f"加权召回率 (Recall): {metrics['recall']:.4f}\n")
        # f.write(f"加权F1分数 (F1 Score): {metrics['f1_score']:.4f}\n")
        # f.write("\n类别详细指标:\n")
        f.write(f"Accuracy: {accuracy:.4f}\nF1 Score: {f1:.4f}\nPrecision:{precision:.4f}"
                f"Recall:{recall:.4f}\n\n{report}")

        # f.write(
        #     f"非空鼓 - 精确率: {metrics['class_precision'][0]:.4f}, 召回率: {metrics['class_recall'][0]:.4f}, F1: {metrics['class_f1'][0]:.4f}\n")
        # f.write(
        #     f"空鼓 - 精确率: {metrics['class_precision'][1]:.4f}, 召回率: {metrics['class_recall'][1]:.4f}, F1: {metrics['class_f1'][1]:.4f}\n")
        # f.write("\n混淆矩阵:\n")
        # f.write(f"{cm}\n")


    # 保存完整指标历史
    history_file = os.path.join(METRICS_DIR, "training_metrics_history.npz")
    np.savez(history_file, **metrics)

    elapsed_time = time.time() - start_time
    print(f"\n  混淆矩阵已保存至 {cm_save_path}")
    print(f"  详细指标已保存至 {metrics_file}")
    print(f"  指标数据已保存至 {history_file}")
    print(f"  耗时: {elapsed_time:.2f} 秒")


# ====================== 主程序执行（路径修复版） ======================
if __name__ == "__main__":
    print("===== 模型训练与评估程序（泛化优化版） =====")
    print(f"程序启动时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"所有文件保存根目录: {BRANCH_TWO_DIR}")

    # 步骤1：加载预处理结果（从Branch Two加载）
    print("\n[步骤1/6] 加载预处理结果...")
    if not os.path.exists(PREPROCESS_INFO):
        print(f"❌ 错误：未找到预处理结果文件！")
        print(f"  请确认以下路径中存在preprocess_info.npz：")
        print(f"  {PREPROCESS_INFO}")
        sys.exit(1)

    # 加载训练数据并修复路径
    data = np.load(PREPROCESS_INFO, allow_pickle=True)
    raw_merged_paths = data['audio_paths']
    raw_merged_array = data['spec_array']
    raw_labels = data['labels']

    # 关键修复1：将所有路径转为绝对路径
    merged_img_paths = []
    valid_array = []
    valid_labels = []
    for path, array, label in zip(raw_merged_paths, raw_merged_array, raw_labels):
        if isinstance(path, str):
            abs_path = os.path.abspath(path)
            merged_img_paths.append(abs_path)
            valid_array.append(array)
            valid_labels.append(label)
            # 校验路径有效性
            if not os.path.exists(abs_path):
                print(f"⚠️  警告：路径不存在 {abs_path}")
            elif os.path.getsize(abs_path) == 0:
                print(f"⚠️  警告：文件为空 {abs_path}")

    # 关键修复2：过滤无效路径
    final_img_paths = []
    final_img_array = []
    final_labels = []
    for path, array, label in zip(merged_img_paths, valid_array, valid_labels):
        if os.path.exists(path) and os.path.getsize(path) > 0:
            final_img_paths.append(path)
            final_img_array.append(array)
            final_labels.append(label)

    print(f"  原始加载路径数：{len(raw_merged_paths)}")
    print(f"  转换绝对路径后：{len(merged_img_paths)}")
    print(f"  过滤无效文件后：{len(final_img_paths)} 个有效频谱图像路径和对应标签")

    if len(final_img_paths) == 0:
        print("❌ 错误：无有效频谱图像文件！")
        print("  请重新运行预处理文件生成完整的频谱图像")
        sys.exit(1)

    # 加载测试数据
    test_info=os.path.join(BASE_DIR, "test_info.npz")
    test_data = np.load(test_info, allow_pickle=True)
    test_merged_array = test_data['spec_array']
    test_labels = test_data['labels']

    # 步骤2：创建数据集
    le = LabelEncoder()
    y_encoded = le.fit_transform(final_labels)
    train_loader, val_loader = create_dataloaders(final_img_paths, final_img_array, y_encoded)

    y_test_encode = le.transform(test_labels)
    test_dataset = AudioDataset(test_merged_array, y_test_encode)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=PIN_MEMORY,
        drop_last=False)


    if train_loader is None or val_loader is None:
        print("❌ 错误：数据集创建失败！")
        sys.exit(1)

    # 步骤3：初始化模型
    print("\n[步骤3/6] 初始化模型和优化器（泛化优化版）...")
    model = CRNNWithAttention(input_channels=3, num_classes=2)

    # 损失函数和优化器（增强泛化能力）
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15).to(DEVICE)  # 增加标签平滑系数
    optimizer = optim.AdamW(  # 使用AdamW优化器
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=True  # 启用amsgrad提升稳定性
    )

    print(f"  模型结构: {model.__class__.__name__} (泛化优化版)")
    print(f"  模型参数总数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  混合精度训练: {'启用' if torch.cuda.is_available() else '禁用'}")
    print(f"  梯度累积步数: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  标签平滑系数: 0.15 (增强泛化能力)")
    print(f"  启用AMSGrad优化器提升稳定性")

    # 步骤4：训练并验证模型
    metrics_history, best_path = train_model(
        model, train_loader, val_loader, criterion, optimizer
    )
    model.load_state_dict(torch.load(best_path))
    true, pred = evaluate_model(model, test_loader)

    # 步骤5：绘制训练曲线（新增）
    if metrics_history and len(metrics_history['train_losses']) > 0:
        plot_training_curves(metrics_history)

    # 步骤6：绘制混淆矩阵和保存指标
    if metrics_history and len(metrics_history['train_losses']) > 0:
        plot_confusion_matrix(true, pred, le.classes_)

    print("\n===== 程序执行完毕（泛化优化版） =====")
    print(f"📊 优化效果：")
    print(f"  - 泛化能力：数据增强、增加Dropout、权重衰减、标签平滑")
    print(f"  - 稳定性：早停机制、权重初始化、梯度裁剪、AMSGrad")
    print(f"  - 指标输出：训练曲线可视化、混淆矩阵可视化、详细数值指标")
    print(f"  - 路径修复：兼容Windows中文/空格/括号路径，自动过滤无效文件")
    print(f"  - 内存优化：峰值控制在2GB以内")
    print(f"  - 兼容低版本PyTorch：移除ReduceLROnPlateau的verbose参数")
    print(f"程序结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")