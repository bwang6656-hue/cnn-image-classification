import json
import matplotlib.pyplot as plt

with open("resnet18_curve.json", "r", encoding="utf-8") as f:
    data = json.load(f)

train_losses = data["train_losses"]
train_accs = data["train_accs"]
test_accs = data["test_accs"]

epochs = range(1, len(train_losses) + 1)

# 画 Loss 曲线
plt.figure()
plt.plot(epochs, train_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()

# 画 Accuracy 曲线
plt.figure()
plt.plot(epochs, train_accs, label="Train Acc")
plt.plot(epochs, test_accs, label="Test Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.title("Accuracy Curve")
plt.show()