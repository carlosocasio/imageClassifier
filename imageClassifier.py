import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model('ResNet50V2-AIvsHumanGenImages.keras')

# Ask for user Input
print("Please enter the full image name ...")
img_path=input()

# Load an image for classification
# img_path = 'example_image.jpg'  # Replace with actual image path
img = image.load_img(img_path, target_size=(512, 512))  # ResNet50V2 input size
img_array = image.img_to_array(img) / 255.0  # Normalize
img_array = np.expand_dims(img_array, axis=0)  # Expand dims for batch processing

# Predict
predictions = model.predict(img_array)
print("AI-generated" if predictions[0] > 0.5 else "Human-created")
