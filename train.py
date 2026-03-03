import torch
import torch.nn as nn
import torch.optim as optim#优化器，提供了多种优化算法，如SGD、Adam等，用于更新模型参数以最小化损失函数
from torchvision import datasets, transforms#计算机视觉库，提供了常用的数据集和数据增强方法
from torch.utils.data import DataLoader#数据加载器，提供了批量加载数据、打乱数据等功能，方便训练和测试模型
from models.simple_cnn import SimpleCNN
from tqdm import tqdm#进度条

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
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

        train_acc = 100. * correct / total
        print(f"[Train] Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%")

        # 简单验证
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = outputs.max(1)
                total += labels.size(0)
                correct += preds.eq(labels).sum().item()
        test_acc = 100. * correct / total
        print(f"[Test ] Epoch {epoch+1}, Acc: {test_acc:.2f}%")

    torch.save(model.state_dict(), "simple_cnn_cifar10.pth")
    print("Model saved to simple_cnn_cifar10.pth")

if __name__ == "__main__":
    train()