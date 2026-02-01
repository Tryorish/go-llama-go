import torch
import sys
import subprocess

print("=" * 50)
print("环境检查报告")
print("=" * 50)

# Python版本
print(f"Python版本: {sys.version}")

# PyTorch信息
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"设备数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  设备 {i}: {torch.cuda.get_device_name(i)}")
    
    # 测试内存
    print(f"当前设备内存分配: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"当前设备缓存: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

# 系统信息
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    print("\nGPU状态:")
    print(result.stdout[:500])  # 只显示前500字符
except:
    print("无法运行nvidia-smi")

print("=" * 50)