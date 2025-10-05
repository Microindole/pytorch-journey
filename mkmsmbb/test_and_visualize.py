import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os

# --- 1. 设置 ---

# 指定模型路径
MODEL_PATH = 'best_classifier_model.pth'

# 定义类别名称
# 重要：这里的顺序必须和你训练时torch.utils.data.ImageFolder的顺序一致
# ImageFolder会按文件夹名称的字母顺序排序，所以 'digimon' 在 'pokemon' 前面
CLASS_NAMES = ['digimon', 'pokemon']

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义用于测试图片的图像变换
# 注意：必须和训练时的验证集(val)变换保持一致！
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# --- 2. 功能函数 ---

def load_model(model_path, num_classes):
    """加载训练好的模型和权重"""
    print("正在加载模型...")
    # 初始化与训练时相同的模型结构 (ResNet18)
    model = models.resnet18()

    # 获取全连接层的输入特征数
    num_ftrs = model.fc.in_features

    # 重构最后一层以匹配我们的模型
    model.fc = torch.nn.Linear(num_ftrs, num_classes)

    # 加载状态字典 (权重)
    # map_location=device 确保了即使模型是在GPU上训练的，也能在CPU上加载
    model.load_state_dict(torch.load(model_path, map_location=device))

    # 将模型切换到评估模式 (这会关闭dropout等)
    model.eval()

    # 将模型移动到指定设备
    model = model.to(device)
    print("模型加载完成！")
    return model


def predict(model, image_path, transforms, class_names):
    """对单张图片进行预测"""
    # 打开图片并确保是RGB格式
    img = Image.open(image_path).convert('RGB')

    # 对图片应用变换，并增加一个批次维度 (unsqueeze(0))
    img_tensor = transforms(img).unsqueeze(0).to(device)

    # 使用 torch.no_grad() 来禁用梯度计算，节省资源
    with torch.no_grad():
        outputs = model(img_tensor)
        # 将模型的输出 logits 转换为概率
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        # 获取概率最高的类别及其置信度
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class = class_names[predicted_idx.item()]
    confidence_score = confidence.item()

    return predicted_class, confidence_score


def visualize_prediction(image_path, prediction, confidence):
    """可视化图片和预测结果"""
    img = Image.open(image_path)

    plt.imshow(img)
    plt.axis('off')  # 关闭坐标轴

    # 设置标题，以百分比格式显示置信度
    title = f"预测结果: {prediction}\n置信度: {confidence:.2%}"
    plt.title(title, fontsize=16)

    # 显示图像窗口
    plt.show()


# --- 3. 主程序入口 ---
if __name__ == '__main__':
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="使用训练好的模型测试单张图片并可视化结果。")
    parser.add_argument('-i', '--image', type=str, required=True, help="需要测试的图片路径。")
    args = parser.parse_args()

    # 检查模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"错误：模型文件 '{MODEL_PATH}' 未找到。请确保它和脚本在同一目录下。")
        exit()

    # 检查图片文件是否存在
    if not os.path.exists(args.image):
        print(f"错误：图片文件 '{args.image}' 未找到。")
        exit()

    # 1. 加载模型
    model = load_model(MODEL_PATH, len(CLASS_NAMES))

    # 2. 进行预测
    prediction, confidence = predict(model, args.image, test_transforms, CLASS_NAMES)

    # 3. 在终端打印结果
    print("-" * 30)
    print(f"图片路径: {args.image}")
    print(f"预测类别: {prediction}")
    print(f"置信度: {confidence:.4f}")
    print("-" * 30)

    # 4. 可视化结果
    visualize_prediction(args.image, prediction, confidence)