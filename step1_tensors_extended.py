import torch
import numpy as np

# 从一个 Python 的 list 创建 Tensor
data = [[1,2,3],[4,5,6],[7,8,9]]
x_data = torch.tensor(data)
print(f'tensor from list:\n{x_data}\n')

# 从一个 NumPy 数组创建 Tensor
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(f'tensor from numpy:\n{x_np}\n')

# 创建一个和 x_data 形状相同，元素全为1的 Tensor
x_ones = torch.ones_like(x_data)
print(f'tensor from ones:\n{x_ones}\n')

# 创建一个和 x_data 形状相同，元素为随机数的 Tensor
x_rand = torch.rand_like(x_data, dtype=torch.float) # 覆盖 dtype 为浮点数
print(f'tensor from random:\n{x_rand}\n')

shape = (2,3,)
random_tensor = torch.randn(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f'tensor from tensor of shape {shape}:\n{random_tensor}\n')

tensor = torch.rand(3,4) # 直接创建一个 3x4 的随机浮点型 Tensor

print(f'shape of tensor:\n{tensor.shape}\n')
print(f'datatype of tensor:\n{tensor.dtype}\n')
print(f'device of tensor:\n{tensor.device}\n')

# 检查 CUDA (GPU支持) 是否可用
if torch.cuda.is_available():
    tensor_gpu = tensor.to('cuda')
    print(f'tensor from cuda:\n{tensor_gpu.device}\n')

tensor = torch.ones(4,4)
tensor[:,1] = 0  # 将第1列 (从0开始数) 的所有元素设置为0
print(f'tensor after slicing:\n{tensor}\n')

# 沿着维度1 (列) 将三个 tensor 拼接起来
t1 = torch.cat([tensor,tensor,tensor],dim=1)
print(f'tensor after concatenation:\n{t1}\n')

# 矩阵乘法
mat_mul = tensor.matmul(tensor.T)
mal_mul_alt = tensor @ tensor.T # @ 是矩阵乘法的简写
print(f'tensor after matrix multiplication:\n{mat_mul}\n')

# 逐元素乘法
elem_mul = tensor.mul(tensor)
elem_mul_alt = tensor * tensor # * 是逐元素乘法的简写
print(f'tensor after element-wise multiplication:\n{elem_mul}\n')

# 聚合操作
agg = tensor.sum()
agg_item = agg.item()
print(f'Sum as a tensor: {agg}, Sum as a Python number: {agg_item}\n')

print('-----------------------------------------')

print("--- 扩展练习 ---")

# 1. 改变形状 (Reshaping)
original_tensor = torch.arange(12)
print(f"Original tensor: {original_tensor}")
reshaped_tensor = original_tensor.reshape(3, 4)
print(f"Reshaped 3x4 tensor:\n{reshaped_tensor}\n")

# 2. 广播机制 (Broadcasting)
a = torch.arange(3).reshape(3, 1)
b = torch.arange(2).reshape(1, 2)
print(f"Tensor a (shape {a.shape}):\n{a}")
print(f"Tensor b (shape {b.shape}):\n{b}")
c = a + b
print(f"a + b after broadcasting (shape {c.shape}):\n{c}\n")

# 3. 就地操作 (In-place operations)
tensor_to_modify = torch.ones(2, 2)
print(f"Original tensor to modify:\n{tensor_to_modify}")
tensor_to_modify.add_(torch.ones(2, 2))
print(f"Tensor after in-place add_():\n{tensor_to_modify}\n")