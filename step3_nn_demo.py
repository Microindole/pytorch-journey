import torch
from torch import nn


# --- 第1部分: 定义一个神经网络 ---

# 在 PyTorch 中，所有的神经网络模型都应该继承自 nn.Module 类。
# 这就像一个模板，为我们提供了很多内置的功能（如参数追踪）。
class SimpleNet(nn.Module):

    # 1. 构造函数 __init__()
    # 这里的任务是“声明”你的网络中会用到哪些层。
    # 把所有需要用到且包含可学习参数的层都在这里定义好。
    def __init__(self, input_size, hidden_size, output_size):
        # 必须首先调用父类 nn.Module 的构造函数
        super(SimpleNet, self).__init__()

        # 定义第一个线性层 (也叫全连接层)
        # nn.Linear(in_features, out_features)
        # in_features: 输入Tensor的特征维度
        # out_features: 该层输出Tensor的特征维度
        # PyTorch会自动创建权重(weight)和偏置(bias)两个Tensor
        self.layer1 = nn.Linear(input_size, hidden_size)

        # 定义激活函数
        # ReLU 是最常用的激活函数之一，它为网络引入非线性。
        # 因为ReLU函数本身没有可学习的参数，所以也可以直接在forward中使用 F.relu()
        self.relu = nn.ReLU()

        # 定义第二个线性层
        # 它的输入维度必须和上一层的输出维度(hidden_size)一致
        self.layer2 = nn.Linear(hidden_size, output_size)

    # 2. 前向传播函数 forward()
    # 这里的任务是“连接”你在__init__中声明的层，定义数据流动的路径。
    # forward函数接收一个输入Tensor `x`，然后返回一个输出Tensor。
    def forward(self, x):
        # 第一步：数据 x 流经第一个线性层
        out1 = self.layer1(x)

        # 第二步：将第一层的输出结果通过ReLU激活函数
        activated_out1 = self.relu(out1)

        # 第三步：将激活后的结果流经第二个线性层
        output = self.layer2(activated_out1)

        return output


# --- 第2部分: 实例化并使用网络 ---

# 1. 定义网络的维度
input_dim = 10  # 输入特征的数量 (比如一个有10个维度的数据点)
hidden_dim = 32  # 隐藏层的大小，这个值可以自己定，是超参数
output_dim = 1  # 输出的维度 (比如预测一个单一的数值)

# 2. 实例化我们刚刚定义的网络
# 就像创建一个普通 Python 对象一样
model = SimpleNet(input_dim, hidden_dim, output_dim)

# 3. 打印网络结构
# 因为我们的类继承了 nn.Module，所以可以直接 print(model)，
# PyTorch 会为我们格式化好一个清晰的结构图。
print("--- Model Structure ---")
print(model)
print("\n")

# 4. 查看模型的所有参数
# nn.Module 的一个强大功能是 .parameters() 方法，
# 它可以自动追踪你在 __init__ 中定义的所有 nn.Module 子类（如nn.Linear）的参数。
print("--- Model Parameters ---")
for name, param in model.named_parameters():
    # .named_parameters() 返回参数的名称和参数本身
    # param 是一个Tensor，包含了权重或偏置的数值
    print(f"Parameter Name: {name}")
    print(f"   - Shape: {param.shape}")
    print(f"   - Requires Grad: {param.requires_grad}")
print("\n")

# 5. 测试一次前向传播
# 创建一个假的输入数据，模拟一个批次(batch)的数据
# batch_size=64, 每个样本有 input_dim=10 个特征
dummy_input = torch.randn(64, input_dim)

# 将数据喂给模型，这会自动调用模型的 forward() 方法
# 即 model(dummy_input) 等价于 model.forward(dummy_input)
prediction = model(dummy_input)

print("--- Forward Pass Test ---")
print(f"Shape of dummy input: {dummy_input.shape}")
print(f"Shape of prediction output: {prediction.shape}")