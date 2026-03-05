import torch
import torch.nn as nn
import torch.optim as optim#优化器，提供了多种优化算法，如SGD、Adam等，用于更新模型参数以最小化损失函数
from torchvision import datasets, transforms#计算机视觉库，提供了常用的数据集和数据增强方法
from torch.utils.data import DataLoader#数据加载器，提供了批量加载数据、打乱数据等功能，方便训练和测试模型
from models.simple_cnn import SimpleCNN
from tqdm import tqdm#进度条
import json

train_losses = []
train_accs = []
test_accs = []

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 数据增强和预处理，增加数据多样性，提高模型的泛化能力，防止过拟合
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),#随机裁剪32x32的图像，边界填充4像素，增加数据多样性
        transforms.RandomHorizontalFlip(),#随机水平翻转图像
        transforms.ToTensor(),#将图像转换为PyTorch张量，并将像素值从[0, 255]缩放到[0, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),#对图像进行归一化处理，使像素值在[-1, 1]范围内，均值和标准差为0.5
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 加载 CIFAR-10 数据集，包含训练集和测试集，使用定义的变换进行预处理
    train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    #训练集加载器：batch_size = 128（每次喂128张图），shuffle = True = 打乱数据，num_workers = 4 = 4线程读取
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)

    #实例化模型，定义损失函数和优化器，模型输出10类，使用交叉熵损失函数，Adam优化器
    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()#交叉熵损失函数，适用于多分类问题，内置了softmax操作，计算模型输出与真实标签之间的差距
    optimizer = optim.Adam(model.parameters(), lr=1e-3)#Adam优化器，学习率为0.001，适用于大多数情况，能够自适应调整学习率

    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            #将数据移动到设备，准备训练
            images, labels = images.to(device), labels.to(device)

            #清梯度 → 前向 → 算损失 → 反向 → 更参数 → 统计
            optimizer.zero_grad()#清零梯度，准备计算新的梯度
            outputs = model(images)#前向传播，得到模型的输出，即每个类的原始得分（未经过softmax处理）
            loss = criterion(outputs, labels)#计算损失，比较模型输出与真实标签之间的差距，得到一个标量值，表示当前模型的性能
            loss.backward()#反向传播，计算损失函数相对于模型参数的梯度，这些梯度将用于更新模型参数
            optimizer.step()#更新模型参数，根据计算得到的梯度调整模型参数，以最小化损失函数

            #统计训练损失和准确率，running_loss累计损失，preds得到模型预测的类别，total统计总样本数，correct统计正确预测的样本数
            running_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        print(f"[Train] Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%")

        # 简单验证
        model.eval()#切换到评估模式，关闭dropout和batch normalization的训练行为
        correct, total = 0, 0
        #with torch.no_grad()：在评估阶段，使用torch.no_grad()上下文管理器，禁用梯度计算，节省内存和计算资源，因为在评估阶段不需要更新模型参数
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                #outputs.max(1)返回每行的最大值和对应的索引，_接收最大值，preds接收索引，即模型预测的类别
                _, preds = outputs.max(1)
                total += labels.size(0)
                #preds.eq(labels)得到一个布尔张量，表示模型预测是否正确，sum()统计正确预测的数量，item()将结果转换为Python数值
                correct += preds.eq(labels).sum().item()
        test_acc = 100. * correct / total#100.表示将准确率转换为百分比，correct / total计算正确预测的比例
        print(f"[Test ] Epoch {epoch+1}, Acc: {test_acc:.2f}%")
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

    torch.save(model.state_dict(), "simple_cnn_cifar10.pth")
    print("Model saved to simple_cnn_cifar10.pth")

    with open("simplecnn_curve.json", "w", encoding="utf-8") as f:
        json.dump({
            "train_losses": train_losses,
            "train_accs": train_accs,
            "test_accs": test_accs
        }, f, indent=2)

    print("Curve data saved to simplecnn_curve.json")

if __name__ == "__main__":
    train()