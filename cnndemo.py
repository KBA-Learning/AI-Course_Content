import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import os

os.makedirs("cnn_outputs", exist_ok=True)

# Load MNIST
(x_train, _), _ = mnist.load_data()
img = x_train[0] / 255.0

# Save original input
plt.imsave("cnn_outputs/input.png", img, cmap="gray")

# Prepare input
img_input = img.reshape(1,28,28,1)

# Build CNN
model = models.Sequential([
    layers.Input(shape=(28,28,1)),
    layers.Conv2D(4, (3,3), activation="relu", name="conv"),
    layers.MaxPooling2D((2,2), name="pool")
])

# 🔹 Build the model by calling it once (VERY IMPORTANT for Keras 3)
_ = model(img_input)

# Create extractor models
conv_model = models.Model(model.inputs, model.get_layer("conv").output)
pool_model = models.Model(model.inputs, model.get_layer("pool").output)

# Run forward pass
conv_output = conv_model.predict(img_input)
pool_output = pool_model.predict(img_input)

# Save convolution outputs
for i in range(conv_output.shape[-1]):
    plt.imsave(f"cnn_outputs/conv_map_{i+1}.png", conv_output[0,:,:,i], cmap="gray")

# Save pooling outputs
for i in range(pool_output.shape[-1]):
    plt.imsave(f"cnn_outputs/pool_map_{i+1}.png", pool_output[0,:,:,i], cmap="gray")

print("All CNN outputs saved in cnn_outputs/ folder")
