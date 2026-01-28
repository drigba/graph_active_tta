#!/usr/bin/env python
"""
Test script to check Flickr dataset properties
"""
print("Started Flickr dataset test script.")
import torch
from omegaconf import OmegaConf
from graph_al.data.build import get_dataset
from graph_al.data.config import TorchGeometricDataConfig, TorchGeometricDatasetType
from graph_al.data.enum import FeatureNormalization

# Create a simple Flickr config
config = TorchGeometricDataConfig(
    root="datagraph-al/data/reddit",
    torch_geometric_dataset=TorchGeometricDatasetType.REDDIT,
    num_classes=7,
    normalize=FeatureNormalization.NONE,
)

# Initialize random generator
generator = torch.Generator().manual_seed(42)

# Load dataset
print("Loading Flickr dataset...")
dataset = get_dataset(config, generator)

print(f"\n=== Flickr Dataset Properties ===")
print(f"Number of nodes: {dataset.base.num_nodes}")
print(f"Number of edges: {dataset.base.num_edges}")
print(f"Number of features: {dataset.base.num_input_features}")
print(f"Number of classes (from config): {config.num_classes}")
print(f"Number of classes (from dataset): {dataset.num_classes}")
print(f"Number of classes (from base): {dataset.base.num_classes}")

print(f"\n=== Label Statistics ===")
print(f"Min label: {dataset.base.labels.min().item()}")
print(f"Max label: {dataset.base.labels.max().item()}")
print(f"Unique labels: {torch.unique(dataset.base.labels).tolist()}")
print(f"Number of unique labels: {len(torch.unique(dataset.base.labels))}")

print(f"\n=== Feature Statistics (before normalization) ===")
print(f"Feature mean: {dataset.base.node_features.mean().item():.4f}")
print(f"Feature std: {dataset.base.node_features.std().item():.4f}")
print(f"Feature min: {dataset.base.node_features.min().item():.4f}")
print(f"Feature max: {dataset.base.node_features.max().item():.4f}")
print(f"Feature L2 norm (avg): {dataset.base.node_features.norm(dim=1).mean().item():.4f}")

# Check after dataset transform
print(f"\n=== After Dataset Transform ===")
print(f"Data feature mean: {dataset.data.x.mean().item():.4f}")
print(f"Data feature std: {dataset.data.x.std().item():.4f}")
print(f"Data feature L2 norm (avg): {dataset.data.x.norm(dim=1).mean().item():.4f}")

print(f"\n=== Class Distribution ===")
for i in range(dataset.num_classes):
    count = (dataset.base.labels == i).sum().item()
    print(f"Class {i}: {count} nodes ({100*count/dataset.base.num_nodes:.2f}%)")

# ============================================
# Train a simple GCN model for sanity check
# ============================================
print("\n" + "="*50)
print("GCN Model Training Sanity Check")
print("="*50)

from graph_al.model.config import GCNConfig
from graph_al.model.build import get_model
from graph_al.active_learning import train_model
from graph_al.model.trainer.config import SGDTrainerConfig
from graph_al.model.trainer.optimizer.config import OptimizerConfigAdam

# Create GCN config with default parameters and SGD trainer
trainer_config = SGDTrainerConfig(
    max_epochs=5,  # Short run for sanity check
    progress_bar=True,
    use_gpu=False,  # Use CPU for testing
    optimizer=OptimizerConfigAdam(lr=0.01, weight_decay=5e-4),
)

gcn_config = GCNConfig(
    hidden_dims=[64],
    dropout=0.8,
    cached=True,
    inplace=False,
    trainer=trainer_config,
)

print(f"\nModel Configuration:")
print(f"  Hidden dims: {gcn_config.hidden_dims}")
print(f"  Dropout: {gcn_config.dropout}")
print(f"  Cached: {gcn_config.cached}")

# Create the model
print("\nInitializing GCN model...")
model = get_model(gcn_config, dataset, generator)
print(f"Model created: {model}")

# Split dataset into train/val/test
print("\nSplitting dataset...")
dataset.split(generator)
print(f"Split sizes - Train: {dataset.data.num_train}, Val: {dataset.data.mask_val.sum().item()}, Test: {dataset.data.mask_test.sum().item()}")

# Acquire some initial training samples (10 per class for balanced training)
print("\nAcquiring initial training samples...")
for class_idx in range(dataset.num_classes):
    class_mask = (dataset.data.y == class_idx) & dataset.data.mask_train_pool
    class_idxs = torch.where(class_mask)[0]
    if len(class_idxs) >= 500:
        to_acquire = class_idxs[:500]
        dataset.add_to_train_idxs(to_acquire)

print(f"Acquired {dataset.data.num_train} training samples")
print(f"Class distribution in training set: {dataset.data.class_counts_train.tolist()}")

# Train for a few epochs as sanity check
print("\nTraining GCN model (short run for sanity check)...")
result = train_model(gcn_config.trainer, model, dataset, generator, acquisition_step=0)

print(f"\n=== Training Results ===")
print(f"Metrics: {result.metrics}")

# ============================================
# Custom Training Loop (without train_model)
# ============================================
print("\n" + "="*50)
print("Custom Training Loop Sanity Check")
print("="*50)

import torch.nn.functional as F
import torch.optim as optim

# Reset the model
print("\nResetting model for custom training loop...")
model = get_model(gcn_config, dataset, generator)

# Setup optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Move data and model to appropriate device
device = torch.device('cpu')
model = model.to(device)
data = dataset.data

# Training loop
print("\nRunning custom training loop...")
num_epochs = 5
model.train()

for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Forward pass
    _,_, out,_= model(data)
    out = F.log_softmax(out, dim=1)

    # Compute loss only on training nodes
    loss = F.nll_loss(out[data.mask_train], data.y[data.mask_train])

    # Backward pass
    loss.backward()
    optimizer.step()

    # Evaluate on train/val every 5 epochs
    if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            _,_, out,_= model(data)
            out = F.log_softmax(out, dim=1)
            pred = out.argmax(dim=1)

            # Calculate accuracy
            train_acc = (pred[data.mask_train] == data.y[data.mask_train]).float().mean()
            val_acc = (pred[data.mask_val] == data.y[data.mask_val]).float().mean()

            print(f"Epoch {epoch+1:3d}: Loss={loss.item():.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        model.train()

# Final evaluation
print("\n=== Final Evaluation ===")
model.eval()
with torch.no_grad():
    _,_, out,_= model(data)
    out = F.log_softmax(out, dim=1)
    pred = out.argmax(dim=1)

    train_acc = (pred[data.mask_train] == data.y[data.mask_train]).float().mean()
    val_acc = (pred[data.mask_val] == data.y[data.mask_val]).float().mean()
    test_acc = (pred[data.mask_test] == data.y[data.mask_test]).float().mean()

    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Val Accuracy:   {val_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")

    # Check for any NaN or Inf values
    has_nan = torch.isnan(out).any()
    has_inf = torch.isinf(out).any()
    print(f"\nOutput has NaN: {has_nan}")
    print(f"Output has Inf: {has_inf}")

print(f"\nAll sanity checks completed successfully!")
