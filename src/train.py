"""
Script for training models for Hybrid On-Device tnGPS experiments.
This script defines model architectures and training functions.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.experiment_config.config import *

class LightweightTNSS(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=LIGHTWEIGHT_HIDDEN_SIZE, output_size=OUTPUT_SIZE):
        super(LightweightTNSS, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.relu(self.fc1(x))
        return self.fc2(out)

class FullScaleLLM(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=FULLSCALE_HIDDEN_SIZE, output_size=OUTPUT_SIZE):
        super(FullScaleLLM, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        return self.fc3(out)

def domain_loss(candidate_output, target_quality=TARGET_QUALITY):
    """
    A simulated domain-specific loss that encourages the candidate's quality (L2 norm)
    to approach a predetermined target value.
    """
    quality = torch.norm(candidate_output, p=2)
    loss = (quality - target_quality) ** 2
    return loss

def fine_tune_model(model, dummy_input, n_steps=FINETUNING_STEPS, lr=FINETUNING_LR):
    """
    A lightweight fine-tuning loop that updates the model based on a domain-specific loss.
    Only selective updates are performed (simulating adapter modules or partial updates).
    """
    model.train()  # Set model in training mode for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for step in range(n_steps):
        optimizer.zero_grad()
        output = model(dummy_input)
        loss = domain_loss(output)
        loss.backward()
        optimizer.step()
    model.eval()  # Return model to evaluation mode after fine-tuning

if __name__ == "__main__":
    print("Training module for Hybrid On-Device tnGPS experiments")
    light_model = LightweightTNSS()
    full_model = FullScaleLLM()
    print(f"LightweightTNSS parameters: {sum(p.numel() for p in light_model.parameters())}")
    print(f"FullScaleLLM parameters: {sum(p.numel() for p in full_model.parameters())}")
