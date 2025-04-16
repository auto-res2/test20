"""
Script to evaluate models for Hybrid On-Device tnGPS experiments.
This script defines evaluation functions and metrics.
"""

import os
import sys
import time
import psutil
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.experiment_config.config import *

sns.set(style="whitegrid")

def run_iterations(model, device, n_iter=PROFILING_ITERATIONS):
    """
    Run a number of iterations with the given model and measure elapsed time and memory usage.
    Returns total runtime and average memory in MB.
    """
    model.to(device)
    model.eval()
    dummy_input = torch.randn(1, INPUT_SIZE).to(device)
    start_time = time.time()
    process = psutil.Process()
    mem_usage = []
    
    for i in range(n_iter):
        current_mem = process.memory_info().rss / (1024 * 1024)
        mem_usage.append(current_mem)
        with torch.no_grad():
            _ = model(dummy_input)
    end_time = time.time()
    total_time = end_time - start_time
    avg_mem = sum(mem_usage) / len(mem_usage)
    return total_time, avg_mem, mem_usage

def evaluate_candidate(output_tensor):
    """
    Calculate a synthetic quality metric based on the L2 norm of the output tensor.
    In this simulation, a higher norm indicates a better candidate.
    """
    quality = torch.norm(output_tensor, p=2).item()
    return quality

def candidate_generation_baseline(model, n_iter=QUALITY_ITERATIONS):
    """
    Generate candidates using the baseline full-scale model.
    """
    model.eval()
    candidate_qualities = []
    dummy_input = torch.randn(1, INPUT_SIZE)
    with torch.no_grad():
        for _ in range(n_iter):
            output = model(dummy_input)
            quality = evaluate_candidate(output)
            candidate_qualities.append(quality)
    return candidate_qualities

def candidate_generation_hybrid(light_model, teacher_model, refine_every=REFINE_EVERY, n_iter=QUALITY_ITERATIONS):
    """
    Generate candidates using the hybrid method:
    Lightweight model proposals that are periodically refined by the teacher (full-scale) model.
    """
    light_model.eval()
    teacher_model.eval()
    candidate_qualities = []
    dummy_input = torch.randn(1, INPUT_SIZE)
    with torch.no_grad():
        for i in range(n_iter):
            output = light_model(dummy_input)
            
            if i % refine_every == 0:
                refined_output = teacher_model(dummy_input)
                output = (output + refined_output) / 2.0
            
            quality = evaluate_candidate(output)
            candidate_qualities.append(quality)
    return candidate_qualities

def candidate_generation_with_finetuning(light_model, teacher_model, refine_every=REFINE_EVERY, n_iter=QUALITY_ITERATIONS):
    """
    Generate candidates using the hybrid method with the addition of on-device fine-tuning
    based on simulated domain feedback.
    """
    from src.train import fine_tune_model  # Import here to avoid circular imports
    
    candidate_qualities = []
    dummy_input = torch.randn(1, INPUT_SIZE)
    
    for i in range(n_iter):
        light_model.eval()
        with torch.no_grad():
            output = light_model(dummy_input)
        
        quality = evaluate_candidate(output)
        candidate_qualities.append(quality)
        
        if i % refine_every == 0:
            teacher_model.eval()
            with torch.no_grad():
                refined_output = teacher_model(dummy_input)
            output = (output + refined_output) / 2.0
        
        if quality < QUALITY_THRESHOLD:
            fine_tune_model(light_model, dummy_input, n_steps=FINETUNING_STEPS, lr=FINETUNING_LR)
            
    return candidate_qualities

if __name__ == "__main__":
    print("Evaluation module for Hybrid On-Device tnGPS experiments")
