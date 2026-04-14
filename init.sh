#!/bin/bash

echo "===================================="
echo "🛠️ Initializing the project"
echo "===================================="

# Generate datasets and reference models
echo "===================================="
echo "🚀 Starting dataset and model setup"
echo "===================================="

echo "[1/4] 🧠 Generating MNIST dataset and model..."
python -m init.mnist
echo "✅ MNIST generation complete."

echo "[2/4] 🫁 Generating ResNet50 for Pneumonia detection..."
python -m init.resnet50-pneumonia
echo "✅ ResNet50 Pneumonia generation complete."

echo "[3/4] Generating VGG16 for cifar10..."
python -m init.vgg16
echo "✅ VGG16 cifar10 complete."

echo "[4/4] Generating Vision Transformer for cifar10..."
python -m init.vision_transformer
echo "✅ Vision Transformer cifar10 complete."

# Generate z_0 vectors
echo "===================================="
echo "🎲 Generating z_0 vectors (various dimensions)"
echo "===================================="

for i in 2 4 8 16 32 64;
do
    python -m init.z_0_generator --z_dim $i
done

echo "===================================="
echo "🏁 All tasks completed successfully!"
echo "===================================="