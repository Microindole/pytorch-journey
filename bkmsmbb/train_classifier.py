import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import os
import time
import copy
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# --- 1. 设置超参数和设备 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据集路径 (请根据你的实际路径修改)
data_dir = '../dataset/'

# 模型参数
batch_size = 32
num_epochs = 25
learning_rate = 0.001

# --- 2. 数据预处理和加载 ---
# 定义数据增强和转换
# 训练集使用更复杂的数据增强，验证集则进行简单的resize和标准化
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),  # 随机裁剪到224x224
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),  # 缩放到256
        transforms.CenterCrop(224),  # 中心裁剪到224x224
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 使用ImageFolder加载数据
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

print(f"Classes: {class_names}")
print(f"Training data size: {dataset_sizes['train']}")
print(f"Validation data size: {dataset_sizes['val']}")


# --- 3. 定义训练模型的函数 ---
def train_model(model, criterion, optimizer, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # 记录训练过程中的损失和准确率
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 每个epoch都有一个训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()  # 设置模型为评估模式

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 梯度清零
                optimizer.zero_grad()

                # 前向传播
                # 只在训练阶段开启梯度计算
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 只在训练阶段进行反向传播和优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计损失和准确率
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # 记录历史数据
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 如果验证集准确率是历史最佳，则保存模型权重
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'best_classifier_model.pth')
                print("Saved Best Model!")

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model, history


# --- 4. 加载预训练模型并修改最后一层 ---
# 我们使用ResNet18，它在性能和速度之间取得了很好的平衡
model_ft = models.resnet18(weights='IMAGENET1K_V1')

# 冻结所有预训练的层
for param in model_ft.parameters():
    param.requires_grad = False

# 获取全连接层的输入特征数
num_ftrs = model_ft.fc.in_features
# 重新定义最后一层，使其输出为我们的类别数（2类）
# 只有这一层是可训练的
model_ft.fc = nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to(device)

# --- 5. 定义损失函数和优化器 ---
criterion = nn.CrossEntropyLoss()
# 只优化我们修改过的全连接层
optimizer_ft = optim.Adam(model_ft.fc.parameters(), lr=learning_rate)

# --- 6. 开始训练 ---
if __name__ == '__main__':
    model_ft, history = train_model(model_ft, criterion, optimizer_ft, num_epochs=num_epochs)

    # 可视化训练过程
    epochs = range(num_epochs)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    # tensor to list for plotting
    train_acc_list = [acc for acc in history['train_acc']]
    val_acc_list = [acc for acc in history['val_acc']]
    plt.plot(epochs, train_acc_list, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc_list, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


# --- 7. 定义预测函数 ---
def predict_image(image_path, model, class_names):
    """
    使用训练好的模型预测单张图片
    """
    model.eval()  # 设置为评估模式

    # 图像预处理
    img = Image.open(image_path).convert('RGB')
    transform = data_transforms['val']  # 使用验证集的变换
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class = class_names[predicted_idx.item()]

    return predicted_class, confidence.item()

# 示例：使用训练好的模型进行预测
# 在训练结束后，可以取消下面的注释来测试一张图片
# if __name__ == '__main__':
#     # 加载我们保存的最佳模型
#     model_to_predict = models.resnet18()
#     num_ftrs = model_to_predict.fc.in_features
#     model_to_predict.fc = nn.Linear(num_ftrs, len(class_names))
#     model_to_predict.load_state_dict(torch.load('best_classifier_model.pth'))
#     model_to_predict = model_to_predict.to(device)

#     # 替换成你自己的图片路径
#     test_image_path = './path/to/your/test_image.jpg'
#     predicted_class, confidence = predict_image(test_image_path, model_to_predict, class_names)
#     print(f"The image is predicted as: {predicted_class} with confidence {confidence:.2f}")