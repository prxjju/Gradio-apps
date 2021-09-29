# Import the neccessary libraries
import gradio as gr
import tensorflow as tf
import numpy as np
import requests

# Load the model
inception_net = tf.keras.applications.InceptionV3() 

# Download human-readable labels for ImageNet.
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")

# Function to preprocess and classify image
def classify_image(inp):
  inp = inp.reshape((-1, 299, 299, 3))
  inp = tf.keras.applications.inception_v3.preprocess_input(inp)
  prediction = inception_net.predict(inp).flatten()
  return {labels[i]: float(prediction[i]) for i in range(1000)}

# Set the gradio inoput and outputs
image = gr.inputs.Image(shape=(299, 299))
label = gr.outputs.Label(num_top_classes=3)

# Launch the interface
gr.Interface(fn=classify_image, inputs=image, outputs=label, capture_session=True).launch()