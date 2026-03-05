import json
import matplotlib.pyplot as plt

with open("resnet18_curve.json", "r", encoding="utf-8") as f:
    resnet = json.load(f)

with open("simplecnn_curve.json", "r", encoding="utf-8") as f:
    cnn = json.load(f)

epochs_resnet = range(1, len(resnet["test_accs"]) + 1)
epochs_cnn = range(1, len(cnn["test_accs"]) + 1)

plt.figure()
plt.plot(epochs_resnet, resnet["test_accs"], label="ResNet18 (Transfer Learning)")
plt.plot(epochs_cnn, cnn["test_accs"], label="Simple CNN (From Scratch)")
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy (%)")
plt.legend()
plt.title("Test Accuracy Comparison")
plt.show()