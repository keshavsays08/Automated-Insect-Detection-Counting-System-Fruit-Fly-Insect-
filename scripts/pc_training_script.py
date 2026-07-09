#!/usr/bin/env python3
"""
Fruit Fly Detection - YOLOv8 Training Script (PC)
Train on your PC, then transfer model to Raspberry Pi
"""

from ultralytics import YOLO
from roboflow import Roboflow
import torch

print("="*60)
print("FRUIT FLY DETECTION - TRAINING ON PC")
print("="*60)

# Step 1: Download dataset from Roboflow
print("\nStep 1: Downloading dataset from Roboflow...")
rf = Roboflow(api_key="YOUR_API_KEY_HERE")  # Replace with your API key
project = rf.workspace("YOUR_WORKSPACE").project("fruit-fly-detection-2rfnl")  # Replace
dataset = project.version(1).download("yolov8")

print(f"✓ Dataset downloaded to: {dataset.location}")

# Step 2: Load YOLOv8 model
print("\nStep 2: Loading YOLOv8 nano model...")
model = YOLO('yolov8n.pt')  # Nano model - fastest, good for Pi

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✓ Using device: {device}")
if device == 'cuda':
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

# Step 3: Train the model
print("\nStep 3: Starting training...")
print("This will take 30-60 minutes on GPU, 2-4 hours on CPU")
print("-"*60)

results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=100,           # Number of training cycles
    imgsz=640,           # Image size
    batch=16,            # Batch size (reduce to 8 if out of memory)
    device=device,       # Use GPU if available
    patience=20,         # Stop early if no improvement
    save=True,
    plots=True,
    
    # Project organization
    project='fruitfly_models',
    name='run1',
    
    # Optimization
    optimizer='AdamW',
    lr0=0.01,
    
    verbose=True
)

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)

# Step 4: Validate the model
print("\nStep 4: Validating model performance...")
metrics = model.val()

print("\n" + "="*60)
print("MODEL PERFORMANCE METRICS:")
print("="*60)
print(f"mAP@50:    {metrics.box.map50:.3f}  (Higher is better, >0.80 is good)")
print(f"mAP@50-95: {metrics.box.map:.3f}   (Higher is better, >0.60 is good)")
print(f"Precision: {metrics.box.mp:.3f}    (How many detections are correct)")
print(f"Recall:    {metrics.box.mr:.3f}    (How many flies were found)")
print("="*60)

# Step 5: Test on sample images
print("\nStep 5: Testing on sample images...")
test_results = model.predict(
    source=f"{dataset.location}/test/images",
    save=True,
    conf=0.5,
    project='fruitfly_models/predictions'
)
print(f"✓ Predictions saved to: fruitfly_models/predictions")

# Step 6: Export for Raspberry Pi
print("\nStep 6: Exporting model for Raspberry Pi deployment...")
model.export(format='torchscript')
print("✓ Model exported to TorchScript format")

print("\n" + "="*60)
print("ALL DONE! 🎉")
print("="*60)
print("\nModel saved at:")
print(f"  Best weights: fruitfly_models/run1/weights/best.pt")
print(f"  Last weights: fruitfly_models/run1/weights/last.pt")
print("\nNext steps:")
print("1. Check metrics above - mAP@50 should be >0.80")
print("2. Review predictions in: fruitfly_models/predictions/")
print("3. Copy best.pt to Raspberry Pi")
print("4. Test on real fruit flies!")
print("="*60)
