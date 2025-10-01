# a_pytorch_journey/step2_autograd_demo.py (Corrected Version)

import torch

# --- 第1部分: 梯度的基本计算 ---

print("--- 1. 梯度的基本计算 ---")
x = torch.ones(2, 2, requires_grad=True)
print(f"x:\n{x}\n")

y = x + 2
print(f"y:\n{y}\n")
print(f"y.grad_fn: {y.grad_fn}\n")

z = y * y * 3
out = z.mean()
print(f"z:\n{z}\n")
print(f"out (这是一个标量): {out}\n")

# 计算梯度
# 魔法发生在这里！对一个标量（通常是损失函数loss）调用 .backward()
# PyTorch会从 out 开始，沿着计算图反向传播，计算出 out 对于 x 的梯度
out.backward()

# 查看梯度
# 计算出的梯度 d(out)/dx 会被累加到 x.grad 属性中
# 数学推导:
# out = (1/4) * Σ z_i = (1/4) * Σ 3 * y_i^2 = (1/4) * Σ 3 * (x_i+2)^2
# 对单个元素求导: d(out)/dx_i = (1/4) * 6 * (x_i+2) = 1.5 * (x_i+2)
# 因为我们设置了 x 的所有元素都为 1, 所以 d(out)/dx_i = 1.5 * (1+2) = 4.5
print(f"x 的梯度 (x.grad):\n{x.grad}\n")


# --- 第2部分: 梯度计算的控制 ---

print("--- 2. 梯度计算的控制 ---")

# 方法1：使用 torch.no_grad() 上下文管理器
# 这是推荐的做法
print("在 torch.no_grad() 环境下:")
with torch.no_grad():
    y_no_grad = x + 2
    print(f"y_no_grad.requires_grad: {y_no_grad.requires_grad}\n")
    # 在这个代码块内的所有计算都不会被追踪

print("使用 .detach() 方法:")
y_detached = y.detach()
print(f"y (原始的) .requires_grad: {y.requires_grad}")
print(f"y_detached.requires_grad: {y_detached.requires_grad}\n")

# --- 第3部分: 梯度累加的特性 ---

print("--- 3. 梯度累加的特性 ---")

# 由于在第1部分我们设置了 retain_graph=True，所以现在可以成功地再次调用 backward()
# PyTorch会再次沿着同一个计算图计算梯度，并累加到 .grad 属性上
out.backward()
print("第二次调用 backward() 后的 x.grad:")
print(x.grad)
# 现在的输出应该是 tensor([[9., 9.], [9., 9.]])，因为 4.5 + 4.5 = 9.0

# 在训练循环中，每次更新权重前，必须手动清零梯度
x.grad.zero_()
print("\n调用 x.grad.zero_() 清零后:")
print(x.grad)