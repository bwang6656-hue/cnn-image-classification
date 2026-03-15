import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.resnet18 import get_resnet18
import json

train_losses = []
train_accs = []
test_accs = []

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # 数据增强和预处理，增加数据多样性，提高模型的泛化能力，防止过拟合
    transform_train = transforms.Compose([
        transforms.Resize(224),#将图像调整为224x224，适配ResNet-18的输入尺寸
        # 先在图像周围填充28像素（相当于在原图基础上扩展到280x280），然后随机裁剪出224x224的图像，增加数据多样性
        #经验值padding ≈ 1/8 图像尺寸
        transforms.RandomCrop(224, padding=28),
        transforms.RandomHorizontalFlip(),#随机水平翻转图像，增加数据多样性
        transforms.ToTensor(),
        #使用ImageNet的均值和标准差进行归一化处理，因为我们使用了在ImageNet上预训练的ResNet-18模型，这样可以更好地利用预训练权重
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),#将图像调整为224x224，适配ResNet-18的输入尺寸
        transforms.CenterCrop(224),#在测试阶段，使用中心裁剪来保持图像的主要内容，同时适配ResNet-18的输入尺寸
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])

    train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=4)

    model = get_resnet18(num_classes=10, pretrained=True).to(device)

    # 先冻结全部层
    for param in model.parameters():
        param.requires_grad = False

    # 解冻 layer4
    for param in model.layer4.parameters():
        param.requires_grad = True

    # 解冻 fc
    for param in model.fc.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    #Fine-Tuning 需要 小学习率保护预训练权重。
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )

    #学习率衰减，每5个epoch衰减为原来的0.1倍，帮助模型更好地收敛，防止过拟合
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.1
    )

    epochs = 10
    for epoch in range(epochs):
        model.train()
        correct, total, running_loss = 0, 0, 0.0

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

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        print(f"[Train] Epoch {epoch+1}, Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")

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

        # 记录曲线数据
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        # 学习率衰减
        scheduler.step()

    torch.save(model.state_dict(), "fine_tuning.pth")
    print("Model saved to fine_tuning.pth")

    with open("fine_tuning_curve.json", "w", encoding="utf-8") as f:
        json.dump({
            "train_losses": train_losses,
            "train_accs": train_accs,
            "test_accs": test_accs
        }, f, indent=2)

    print("Curve data saved to fine_tuning_curve.json")

if __name__ == "__main__":
    train()