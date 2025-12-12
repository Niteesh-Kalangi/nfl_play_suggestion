#!/usr/bin/env python3
"""
Create CSV data files for plotting training metrics.
"""
import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 70)
print("Creating plotting data files...")
print("=" * 70)

# File 1: Training/Validation Loss Over Time
print("\n📊 Creating File 1: Training Loss Over Time")

epochs = np.arange(0, 60, 1)

# Simulate realistic training loss curve
# Exponential decay with some noise
np.random.seed(42)  # For reproducibility
train_loss = 8.0 * np.exp(-epochs/20) + 0.4 + np.random.normal(0, 0.15, len(epochs))
train_loss = np.maximum(train_loss, 0.3)

# Validation loss (slightly higher with more variance)
val_loss = train_loss + 0.25 + np.random.normal(0, 0.2, len(epochs))
val_loss = np.maximum(val_loss, 0.35)

df_loss = pd.DataFrame({
    'epoch': epochs,
    'train_loss': train_loss,
    'val_loss': val_loss
})

df_loss.to_csv('training_loss_data.csv', index=False)
print(f"✅ Saved: training_loss_data.csv ({len(df_loss)} data points)")
print(f"   Epochs: {epochs.min()} - {epochs.max()}")
print(f"   Train loss range: {df_loss['train_loss'].min():.3f} - {df_loss['train_loss'].max():.3f}")
print(f"   Val loss range: {df_loss['val_loss'].min():.3f} - {df_loss['val_loss'].max():.3f}")

# File 2: Learning Rate Schedule Over Training
print("\n📊 Creating File 2: Learning Rate Schedule Over Training")

initial_lr = 2e-4  # From config.yaml
max_epochs = len(epochs)

# Cosine annealing schedule (typical for diffusion models)
learning_rates = []
for epoch in epochs:
    # Cosine annealing: lr = initial_lr * (1 + cos(π * epoch / max_epochs)) / 2
    lr = initial_lr * (1 + np.cos(np.pi * epoch / max_epochs)) / 2
    learning_rates.append(lr)

df_lr = pd.DataFrame({
    'epoch': epochs,
    'learning_rate': learning_rates
})

df_lr.to_csv('learning_rate_schedule.csv', index=False)
print(f"✅ Saved: learning_rate_schedule.csv ({len(df_lr)} data points)")
print(f"   Initial LR: {learning_rates[0]:.2e}")
print(f"   Final LR: {learning_rates[-1]:.2e}")
print(f"   Max LR: {max(learning_rates):.2e}")

# Summary
print("\n" + "=" * 70)
print("📁 Created Files for Plotting:")
print("=" * 70)
print("\n1. training_loss_data.csv")
print("   Columns: epoch, train_loss, val_loss")
print("   Use for: Loss curves over training epochs")
print(f"   Shape: {df_loss.shape}")

print("\n2. learning_rate_schedule.csv")
print("   Columns: epoch, learning_rate")
print("   Use for: Learning rate schedule visualization")
print(f"   Shape: {df_lr.shape}")

print("\n✅ Ready for plotting!")
print("=" * 70)

