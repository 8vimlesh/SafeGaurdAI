#!/usr/bin/env python3
"""
CUDA Installation Verification Script
Checks NVIDIA drivers, CUDA toolkit, cuDNN, PyTorch, and GPU capabilities
"""

import subprocess
import sys
import os
import re
import time
from pathlib import Path

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("ERROR: PyTorch not installed. Install with:")
    print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
    sys.exit(1)


def get_nvidia_driver_version():
    """Get NVIDIA driver version from nvidia-smi"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return None


def get_cuda_toolkit_version():
    """Get CUDA toolkit version"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        # Try to get CUDA version from nvcc
        try:
            nvcc_result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if nvcc_result.returncode == 0:
                # Parse version from nvcc output
                match = re.search(r"release\s+([\d.]+)", nvcc_result.stdout)
                if match:
                    return match.group(1)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Fallback: use torch.version.cuda
        return torch.version.cuda if hasattr(torch.version, 'cuda') else None
    except Exception:
        return None


def get_cudnn_version():
    """Get cuDNN version"""
    try:
        if hasattr(torch.backends.cudnn, 'version'):
            return torch.backends.cudnn.version()
        return None
    except Exception:
        return None


def get_pytorch_version():
    """Get PyTorch version"""
    return torch.__version__


def get_gpu_device_name():
    """Get GPU device name"""
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "N/A"
    except Exception as e:
        return f"Error: {str(e)}"


def get_gpu_memory():
    """Get GPU memory info"""
    try:
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            reserved_memory = torch.cuda.memory_reserved(0) / (1024**3)
            allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
            free_memory = total_memory - (allocated_memory)
            return (total_memory, free_memory)
        return (0, 0)
    except Exception:
        return (0, 0)


def get_compute_capability():
    """Get GPU compute capability"""
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            compute_cap = props.major, props.minor
            return f"{compute_cap[0]}.{compute_cap[1]}"
        return "N/A"
    except Exception:
        return "N/A"


def benchmark_matmul(num_ops=100, matrix_size=1000):
    """Benchmark matmul operations on CPU vs GPU"""
    results = {"cpu": None, "gpu": None, "speedup": None}
    
    try:
        # CPU Benchmark
        if True:
            A_cpu = torch.randn(matrix_size, matrix_size)
            B_cpu = torch.randn(matrix_size, matrix_size)
            
            # Warmup
            for _ in range(5):
                _ = torch.matmul(A_cpu, B_cpu)
            
            # Actual benchmark
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()
            for _ in range(num_ops):
                _ = torch.matmul(A_cpu, B_cpu)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            cpu_time = (time.time() - start) * 1000
            results["cpu"] = cpu_time
        
        # GPU Benchmark
        if torch.cuda.is_available():
            A_gpu = torch.randn(matrix_size, matrix_size).cuda()
            B_gpu = torch.randn(matrix_size, matrix_size).cuda()
            
            # Warmup
            for _ in range(5):
                _ = torch.matmul(A_gpu, B_gpu)
            
            # Actual benchmark
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(num_ops):
                _ = torch.matmul(A_gpu, B_gpu)
            torch.cuda.synchronize()
            gpu_time = (time.time() - start) * 1000
            results["gpu"] = gpu_time
            
            # Calculate speedup
            if cpu_time > 0 and gpu_time > 0:
                results["speedup"] = cpu_time / gpu_time
        
        return results
    except Exception as e:
        print(f"Benchmark error: {str(e)}")
        return results


def print_verification_report():
    """Print comprehensive CUDA verification report"""
    
    # Collect information
    nvidia_driver = get_nvidia_driver_version()
    cuda_toolkit = get_cuda_toolkit_version()
    cudnn_version = get_cudnn_version()
    pytorch_version = get_pytorch_version()
    cuda_available = torch.cuda.is_available()
    gpu_name = get_gpu_device_name()
    gpu_memory = get_gpu_memory()
    compute_cap = get_compute_capability()
    
    # Run benchmarks if CUDA is available
    benchmarks = None
    if cuda_available:
        benchmarks = benchmark_matmul(num_ops=100, matrix_size=1000)
    
    # Format output
    print("\n" + "=" * 50)
    print("  CUDA INSTALLATION VERIFICATION")
    print("=" * 50)
    
    # Driver info
    driver_str = nvidia_driver if nvidia_driver else "NOT FOUND"
    print(f"NVIDIA Driver    : {driver_str}")
    
    # CUDA Toolkit
    toolkit_str = cuda_toolkit if cuda_toolkit else "NOT FOUND"
    print(f"CUDA Toolkit     : {toolkit_str}")
    
    # cuDNN
    cudnn_str = f"{cudnn_version}" if cudnn_version else "NOT FOUND"
    print(f"cuDNN            : {cudnn_str}")
    
    # PyTorch
    print(f"PyTorch Version  : {pytorch_version}")
    
    # CUDA Support in PyTorch
    cuda_status = "✓ Enabled" if cuda_available else "✗ Disabled"
    print(f"PyTorch CUDA     : {cuda_status}")
    
    # GPU Device
    print(f"GPU Device       : {gpu_name}")
    
    # GPU Memory
    if gpu_memory[0] > 0:
        total_gb = gpu_memory[0]
        free_gb = gpu_memory[1]
        print(f"GPU Memory       : {total_gb:.1f} GB total, {free_gb:.1f} GB available")
    else:
        print(f"GPU Memory       : N/A")
    
    # Compute Capability
    print(f"Compute Cap      : {compute_cap}")
    
    print("=" * 50)
    
    # Benchmark results
    if benchmarks:
        cpu_time = benchmarks.get("cpu")
        gpu_time = benchmarks.get("gpu")
        speedup = benchmarks.get("speedup")
        
        print("CUDA Computation Test:")
        if cpu_time:
            print(f"  CPU matmul (100x) : {cpu_time:.3f} ms")
        if gpu_time:
            print(f"  GPU matmul (100x) : {gpu_time:.3f} ms")
        if speedup:
            print(f"  Speedup          : {speedup:.2f}x faster")
    
    print("=" * 50)
    
    # Status determination
    status_ok = True
    issues = []
    
    if not nvidia_driver:
        status_ok = False
        issues.append("NVIDIA driver not found — install from nvidia.com")
    
    if not cuda_available:
        status_ok = False
        issues.append("PyTorch CUDA support missing — run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
    
    if gpu_memory[0] > 0 and gpu_memory[1] < 0.5:
        status_ok = False
        issues.append(f"Low GPU memory — only {gpu_memory[1]:.1f} GB available")
    
    if not cudnn_version and cuda_available:
        issues.append("⚠ cuDNN not found (optional but recommended for optimal performance)")
    
    # Print status and issues
    if status_ok and cuda_available:
        print("Status: ✓ READY FOR GPU TRAINING")
    else:
        print("Status: ✗ CUDA NOT PROPERLY CONFIGURED")
    
    if issues:
        print("\nIssues/Recommendations:")
        for issue in issues:
            print(f"  • {issue}")
    
    print("=" * 50 + "\n")
    
    return status_ok


def main():
    """Main entry point"""
    try:
        success = print_verification_report()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
