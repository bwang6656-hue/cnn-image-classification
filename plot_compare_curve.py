import json
import matplotlib.pyplot as plt

with open("resnet18_curve.json", "r", encoding="utf-8") as f:
    resnet = json.load(f)

with open("resnet18_curve_2.json", "r", encoding="utf-8") as f:
    resnet_cifar = json.load(f)

with open("simplecnn_curve.json", "r", encoding="utf-8") as f:
    cnn = json.load(f)

with open("fine_tuning_curve.json", "r", encoding="utf-8") as f:
    resnet_fine_tuning = json.load(f)

epochs_resnet = range(1, len(resnet["test_accs"]) + 1)
epochs_resnet_cifar = range(1, len(resnet_cifar["test_accs"]) + 1)
epochs_cnn = range(1, len(cnn["test_accs"]) + 1)
epochs_fine_tuning = range(1, len(resnet_fine_tuning["test_accs"]) + 1)

plt.figure()
plt.plot(epochs_resnet, resnet["test_accs"], label="ResNet18 (Transfer Learning)")
plt.plot(epochs_resnet_cifar, resnet_cifar["test_accs"], label="ResNet18 (From Scratch)")
plt.plot(epochs_cnn, cnn["test_accs"], label="Simple CNN (From Scratch)")
plt.plot(epochs_fine_tuning, resnet_fine_tuning["test_accs"], label="ResNet18 (Fine Tuning)")
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy (%)")
plt.legend()
plt.title("Test Accuracy Comparison")
plt.show()