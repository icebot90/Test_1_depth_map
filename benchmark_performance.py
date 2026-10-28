"""
Performance benchmark script for measuring latencies, throughput, and memory consumption.
"""

import os
import time
import numpy as np
import torch
import psutil

from integration.combined_inference import CombinedInferencePipeline
from integration.config import PipelineConfig

def run_benchmark():
    process = psutil.Process(os.getpid())
    ram_before = process.memory_info().rss / (1024 * 1024)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    config = PipelineConfig(no_display=True)
    pipeline = CombinedInferencePipeline(config)

    # Warmup
    warmup_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(5):
        pipeline.process_frame(warmup_frame)

    resolutions = [
        ("VGA (640x480)", (480, 640)),
        ("HD (1280x720)", (720, 1280)),
    ]

    print("\n================ BENCHMARK RESULTS ================")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device_name}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    for name, (h, w) in resolutions:
        # Create synthetic test frame with realistic human-like texture
        test_frame = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
        num_runs = 20
        det_times = []
        depth_times = []
        total_times = []

        for _ in range(num_runs):
            _, _, _, _, stats = pipeline.process_frame(test_frame)
            det_times.append(stats["det_ms"])
            depth_times.append(stats["depth_ms"])
            total_times.append(stats["total_ms"])

        avg_det = np.mean(det_times)
        avg_depth = np.mean(depth_times)
        avg_total = np.mean(total_times)
        fps = 1000.0 / avg_total

        print(f"\nResolution: {name}")
        print(f"  Runs:                  {num_runs}")
        print(f"  Avg D-FINE Latency:    {avg_det:.2f} ms")
        print(f"  Avg Depth Latency:     {avg_depth:.2f} ms")
        print(f"  Avg Total Latency:     {avg_total:.2f} ms")
        print(f"  Throughput:            {fps:.2f} FPS")

    ram_after = process.memory_info().rss / (1024 * 1024)
    print(f"\nProcess RAM Usage:       {ram_after:.1f} MB (Delta: {ram_after - ram_before:.1f} MB)")
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"Peak GPU VRAM Allocated: {peak_vram:.1f} MB")
    print("====================================================\n")

if __name__ == "__main__":
    run_benchmark()
