import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("fruit_model.h5")

# IMPORTANT: this must exactly match the folder names/order from training
# Copy the "Classes found:" list that train.py prints, don't guess it
class_names = ["apple", "avocado", "banana", "cherry", "kiwi", "mango",
               "orange", "pinenapple", "strawberries", "watermelon"]

img = image.load_img("test.png", target_size=(100, 100))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # match training rescale

predictions = model.predict(img_array)
predicted_class = class_names[np.argmax(predictions)]
confidence = np.max(predictions) * 100

print(f"Predicted: {predicted_class} ({confidence:.2f}% confidence)")