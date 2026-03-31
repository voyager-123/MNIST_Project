import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time


print("="*60)
print("开始 MNIST 手写数字识别项目")
print(f"PyTorch 版本: {torch.__version__}")
print(f"是否使用GPU: {'是' if torch.cuda.is_available() else '否'} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
print("="*60)

# 设置设备（GPU 或 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 1. 准备数据（使用正规化提升训练效率）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 标准正规化参数
])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 使用 DataLoader 进行高效的批量加载
batch_size = 128  # 增大批量，充分利用GPU
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True if torch.cuda.is_available() else False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True if torch.cuda.is_available() else False)

print(f"\n数据加载完成！")
print(f"训练样本数: {len(train_data)}")
print(f"测试样本数: {len(test_data)}")
print(f"批量大小: {batch_size}")

# 2. 定义改进的神经网络模型（具有隐藏层）
class ImprovedNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 512)    # 第一隐藏层
        self.fc2 = nn.Linear(512, 256)      # 第二隐藏层
        self.fc3 = nn.Linear(256, 10)       # 输出层
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)      # 防止过拟合
    
    def forward(self, x):
        x = x.view(-1, 28*28)  # 将图片展平
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

model = ImprovedNN().to(device)  # 移到指定设备
print(f"\n模型已创建: {model}")
print(f"模型参数数: {sum(p.numel() for p in model.parameters()):,}")

# 3. 设置损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器更高效
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # 学习率调度

# 4. 定义训练函数
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # 移到设备
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# 5. 定义评估函数
def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# 6. 训练模型
num_epochs = 10
print(f"\n>>> 开始训练模型（{num_epochs}个轮次）...")
start_time = time.time()

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    scheduler.step()
    
    print(f"轮次 {epoch+1}/{num_epochs} | "
          f"训练损失: {train_loss:.4f} | 训练精度: {train_acc:.2f}% | "
          f"测试损失: {test_loss:.4f} | 测试精度: {test_acc:.2f}%")

elapsed_time = time.time() - start_time
print(f"\n✅ 训练完成！用时: {elapsed_time:.2f}秒")

# 7. 可视化第一个训练样本
print("\n>>> 正在显示第一个训练样本...")
img, label = train_data[0]
plt.figure(figsize=(4, 4))
plt.imshow(img.squeeze(), cmap='gray')
plt.title(f'标签: {label}')
plt.axis('off')
plt.tight_layout()
plt.show()

# 8. 测试模型在单个样本上的预测
print("\n>>> 进行单个样本预测...")
model.eval()
test_img, true_label = test_data[0]
with torch.no_grad():
    test_img = test_img.unsqueeze(0).to(device)
    output = model(test_img)
    _, pred_label = torch.max(output, 1)
    print(f"真实标签: {true_label}, 预测标签: {pred_label.item()}")

print("\n✅ MNIST 项目优化演示成功完成！")
print("性能提升:")
print("  • 使用GPU加速")
print("  • 批量处理（128个样本）")
print("  • 改进的模型架构（隐藏层 + Dropout）")
print("  • Adam优化器（比SGD更高效）")
print("  • 学习率动态调整")