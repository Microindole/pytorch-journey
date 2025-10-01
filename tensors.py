import torch
import numpy as np
from sympy.physics.units import electronvolt

data = [[1,2,3],[4,5,6],[7,8,9]]
x_data = torch.tensor(data)
print(f'tensor from list:\n{x_data}\n')

np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(f'tensor from numpy:\n{x_np}\n')

x_ones = torch.ones_like(x_data)
print(f'tensor from ones:\n{x_ones}\n')

x_rand = torch.rand_like(x_data,dtype=torch.float)
print(f'tensor from random:\n{x_rand}\n')

shape = (2,3,)
random_tensor = torch.randn(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f'tensor from tensor of shape {shape}:\n{random_tensor}\n')

tensor = torch.rand(3,4)
print(f'shape of tensor:\n{tensor.shape}\n')
print(f'datatype of tensor:\n{tensor.dtype}\n')
print(f'device of tensor:\n{tensor.device}\n')

if torch.cuda.is_available():
    tensor_gpu = tensor.to('cuda')
    print(f'tensor from cuda:\n{tensor_gpu.device}\n')

tensor = torch.ones(4,4)
tensor[:,1] = 0
print(f'tensor after slicing:\n{tensor}\n')

t1 = torch.cat([tensor,tensor,tensor],dim=1)
print(f'tensor after concatenation:\n{t1}\n')

mat_mul = tensor.matmul(tensor.T)

mal_mul_alt = tensor @ tensor.T
print(f'tensor after multiplication:\n{mat_mul}\n')

elem_mul = tensor.mul(tensor)
elem_mul_alt = tensor * tensor
print(f'tensor after multiplication:\n{elem_mul}\n')

agg = tensor.sum()
agg_item = agg.item()
print(f'Sum as a tensor: {agg},Sum as a Python number: {agg_item}\n')


