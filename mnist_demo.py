import torch
import torch.nn as nn 
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path

# Matplotlib 字体设置，避免中文乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("="*60)
print("开始 MNIST 手写数字识别项目")
print(f"PyTorch 版本: {torch.__version__}")
print(f"是否使用GPU: {'是' if torch.cuda.is_available() else '否'} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
print("="*60)

# ==================== 配置参数 ====================
CONFIG = {
    'batch_size': 128,
    'num_epochs': 10,
    'learning_rate': 0.001,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'seed': 42,
    'model_save_path': './models/best_model.pth',
    'patience': 3,  # Early stopping 参数
    'num_workers': 2 if os.name != 'nt' else 0,  # Windows 下设为 0
}

# 设置随机种子以保证可重复性
torch.manual_seed(CONFIG['seed'])

device = CONFIG['device']
print(f"使用设备: {device}")

# ==================== 准备数据 ====================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 标准正规化参数
])

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 使用 DataLoader 进行高效的批量加载
use_pin_memory = torch.cuda.is_available()
train_loader = DataLoader(
    train_data, 
    batch_size=CONFIG['batch_size'], 
    shuffle=True, 
    num_workers=CONFIG['num_workers'], 
    pin_memory=use_pin_memory,
    persistent_workers=False
)
test_loader = DataLoader(
    test_data, 
    batch_size=CONFIG['batch_size'], 
    shuffle=False, 
    num_workers=CONFIG['num_workers'], 
    pin_memory=use_pin_memory,
    persistent_workers=False
)

print(f"\n数据加载完成！")
print(f"训练样本数: {len(train_data)}")
print(f"测试样本数: {len(test_data)}")
print(f"批量大小: {CONFIG['batch_size']}")

# ==================== 定义神经网络模型 ====================
class ImprovedNN(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 512)    # 第一隐藏层
        self.fc2 = nn.Linear(512, 256)      # 第二隐藏层
        self.fc3 = nn.Linear(256, 10)       # 输出层
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm1 = nn.BatchNorm1d(512)  # 批正规化，加速训练
        self.batch_norm2 = nn.BatchNorm1d(256)
    
    def forward(self, x):
        x = x.view(-1, 28*28)  # 将图片展平
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x

model = ImprovedNN().to(device)
print(f"\n模型已创建: {model}")
print(f"模型参数数: {sum(p.numel() for p in model.parameters()):,}")

# ==================== 设置优化器和损失函数 ====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2, verbose=True
)  # 动态调整学习率

# 创建模型保存目录
Path(CONFIG['model_save_path']).parent.mkdir(parents=True, exist_ok=True)

# ==================== 定义训练和评估函数 ====================
def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个轮次"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(output.detach(), 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

def evaluate(model, test_loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

def save_model(model, optimizer, epoch, path):
    """保存模型检查点"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"✓ 模型已保存到: {path}")

def load_model(model, optimizer, path):
    """加载模型检查点"""
    checkpoint = torch.load(path, map_location=CONFIG['device'])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"✓ 模型已加载: {path}")
    return checkpoint['epoch']

# ==================== 训练模型（包含 Early Stopping）====================
print(f"\n>>> 开始训练模型（最多{CONFIG['num_epochs']}个轮次）...")
start_time = time.time()

best_test_loss = float('inf')
patience_counter = 0
history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

for epoch in range(CONFIG['num_epochs']):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['test_loss'].append(test_loss)
    history['test_acc'].append(test_acc)
    
    scheduler.step(test_loss)
    
    # Early Stopping 逻辑
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        patience_counter = 0
        save_model(model, optimizer, epoch, CONFIG['model_save_path'])
    else:
        patience_counter += 1
    
    status = "✓" if patience_counter == 0 else " "
    print(f"{status} 轮次 {epoch+1:2d}/{CONFIG['num_epochs']} | "
          f"训练: 损失={train_loss:.4f}, 精度={train_acc:.2f}% | "
          f"测试: 损失={test_loss:.4f}, 精度={test_acc:.2f}%")
    
    # 如果没有改进，提前停止
    if patience_counter >= CONFIG['patience']:
        print(f"\n⚠ 已连续{CONFIG['patience']}个轮次未改进，触发 Early Stopping")
        break

elapsed_time = time.time() - start_time
print(f"\n✅ 训练完成！总耗时: {elapsed_time:.2f}秒")
print(f"最佳模型已保存到: {CONFIG['model_save_path']}")

# ==================== 加载最佳模型并进行测试 ====================
model.load_state_dict(torch.load(CONFIG['model_save_path'], map_location=device)['model_state_dict'])

# 在测试集上的最终评估
final_test_loss, final_test_acc = evaluate(model, test_loader, criterion, device)
print(f"\n>>> 最佳模型在测试集上的表现:")
print(f"   测试损失: {final_test_loss:.4f}")
print(f"   测试精度: {final_test_acc:.2f}%")

# ==================== 可视化训练过程 ====================
Path('outputs').mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='训练损失 / Train Loss', marker='o')
plt.plot(history['test_loss'], label='测试损失 / Test Loss', marker='s')
plt.xlabel('轮次 / Epoch')
plt.ylabel('损失 / Loss')
plt.title('损失变化曲线 / Loss Curve')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='训练精度 / Train Acc', marker='o')
plt.plot(history['test_acc'], label='测试精度 / Test Acc', marker='s')
plt.xlabel('轮次 / Epoch')
plt.ylabel('精度 (%) / Accuracy')
plt.title('精度变化曲线 / Accuracy Curve')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/loss_accuracy_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# ==================== 可视化预测结果 ====================
print("\n>>> 显示前9个测试样本的预测结果...")
model.eval()
fig, axes = plt.subplots(3, 3, figsize=(8, 8))

with torch.no_grad():
    for i in range(9):
        test_img, true_label = test_data[i]
        input_tensor = test_img.unsqueeze(0).to(device)
        output = model(input_tensor)
        _, pred_label = torch.max(output, 1)
        pred_label = pred_label.item()
        
        ax = axes[i // 3, i % 3]
        ax.imshow(test_img.squeeze(), cmap='gray')
        color = 'green' if pred_label == true_label else 'red'
        ax.set_title(f'真实: {true_label}, 预测: {pred_label}', color=color)
        ax.axis('off')

plt.tight_layout()
plt.savefig('outputs/prediction_grid.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ MNIST 项目优化完成！")
print("\n优化亮点:")
print("  ✓ 配置参数管理 - 易于调整和维护")
print("  ✓ 批正规化层 - 加速收敛")
print("  ✓ 动态学习率调整 - 根据验证损失自动调整")
print("  ✓ Early Stopping - 防止过拟合")
print("  ✓ 模型检查点保存 - 自动保存最佳模型")
print("  ✓ 梯度裁剪 - 稳定训练过程")
print("  ✓ 训练历史记录 - 便于分析")
print("  ✓ 训练可视化 - 直观展示训练效果")