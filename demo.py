import torch

# 1. 检查CUDA是否可用，并设置设备
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"太棒了！我们将在GPU上运行: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("警告：CUDA不可用，我们将使用CPU。")

# 2. 创建一个在CPU上的张量
cpu_tensor = torch.tensor([1, 2, 3])
print(f"\n原始张量在: {cpu_tensor.device}")

# 3. 将张量移动到我们选择的设备 (GPU)
gpu_tensor = cpu_tensor.to(device)
print(f"移动后的张量在: {gpu_tensor.device}")

# 4. 在GPU上执行一个操作
# 这个操作的结果会自动存储在同一个设备上
result_tensor = gpu_tensor + gpu_tensor
print(f"加法运算的结果在: {result_tensor.device}")

# 5. 查看结果
print(f"计算结果: {result_tensor}")

# 如果需要将结果移回CPU（例如，与Numpy库交互或打印）
# result_on_cpu = result_tensor.cpu()