import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Load the model once
model = tf.keras.models.load_model('pneumonia_vgg16.h5')

# 2. Define the core prediction function
def predict_pneumonia(image):
    # a. Preprocess the image to match VGG16 requirements
    if image is None:
        return None
        
    # Convert to RGB (in case user uploads B&W)
    image = image.convert('RGB')
    
    # Resize to 150x150
    image = image.resize((150, 150))
    
    # Convert to array
    x = np.array(image)
    
    # --- THE FIX IS HERE ---
    # Convert from Integers (uint8) to Floats (float32) BEFORE dividing
    x = x.astype('float32')
    
    # Add batch dimension (1, 150, 150, 3)
    x = np.expand_dims(x, axis=0)
    
    # Normalize (0 to 1)
    x /= 255.0

    # b. Predict
    prediction = model.predict(x)[0][0]
    
    # c. Return result
    return {
        "Pneumonia": float(prediction),
        "Normal": float(1 - prediction)
    }

# 3. Create the Gradio Interface
interface = gr.Interface(
    fn=predict_pneumonia,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="AI Pneumonia Detector",
    description="Upload a Chest X-Ray to detect signs of Pneumonia."
)

# 4. Launch
if __name__ == "__main__":
    interface.launch()