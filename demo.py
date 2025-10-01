import torch
print(f'pytorch version: {torch.__version__}')

x = torch.randn(2,3)
print(x)

is_cuda_available = torch.cuda.is_available()
print(is_cuda_available)

if is_cuda_available:
    print(f'cuda version: {torch.version.cuda}')
    print(f'gpu count: {torch.cuda.device_count()}')
    print(f'gpu id: {torch.cuda.current_device()}')
    print(f'gpu name: {torch.cuda.get_device_name(torch.cuda.current_device())}')
    