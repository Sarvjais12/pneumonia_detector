import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# 1. Load the model once
model = tf.keras.models.load_model('pneumonia_vgg16.h5')

# 2. Dynamically find the last convolutional layer for Grad-CAM
last_conv_layer_name = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer_name = layer.name
        break

# 3. Grad-CAM Algorithm
# 3. Grad-CAM Algorithm
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Create a model that maps the input image to the activations of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        outputs = grad_model(img_array)
        
        # THE FIX: Forcefully extract the tensors if Keras 3 wraps them in a list
        last_conv_layer_output = outputs[0] if not isinstance(outputs[0], list) else outputs[0][0]
        preds = outputs[1] if not isinstance(outputs[1], list) else outputs[1][0]

        if pred_index is None:
            pred_index = tf.argmax(preds[0])
            
        # preds is now guaranteed to be a Tensor, so slicing will work perfectly
        class_channel = preds[:, pred_index]

    # Compute gradients
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Pool the gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the feature map
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap safely (preventing divide-by-zero)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val != 0:
        heatmap = heatmap / max_val
        
    return heatmap.numpy()

# 4. Core Prediction & Visualization Function
def predict_and_visualize(image):
    if image is None:
        return None, None
        
    # --- PREPROCESSING ---
    img_rgb = image.convert('RGB')
    img_resized = img_rgb.resize((150, 150))
    x = np.array(img_resized).astype('float32') # The critical uint8 to float32 fix
    x_batch = np.expand_dims(x, axis=0)
    x_norm = x_batch / 255.0

    # --- INFERENCE ---
    prediction = model.predict(x_norm)[0][0]
    confidence = {
        "Pneumonia": float(prediction),
        "Normal": float(1 - prediction)
    }

    # --- GRAD-CAM GENERATION ---
    if last_conv_layer_name is not None:
        heatmap = make_gradcam_heatmap(x_norm, model, last_conv_layer_name)
        
        # Resize heatmap to match original image size
        heatmap_resized = cv2.resize(heatmap, (image.size[0], image.size[1]))
        
        # Convert to RGB colormap (JET)
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_colormap = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        
        # Superimpose the heatmap on original image
        # THE FIX: Changed COLORMAP_RGB2BGR to COLOR_RGB2BGR
        original_img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        superimposed_img = cv2.addWeighted(original_img_cv, 0.6, heatmap_colormap, 0.4, 0)
        
        # Convert back to PIL Image for Gradio
        # THE FIX: Changed COLORMAP_BGR2RGB to COLOR_BGR2RGB
        final_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
        out_image = Image.fromarray(final_img)
    else:
        # Fallback if no conv layer is found
        out_image = image

    return confidence, out_image

# 5. Build the updated UI
interface = gr.Interface(
    fn=predict_and_visualize,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=2, label="Prediction Confidence"), 
        gr.Image(type="pil", label="Grad-CAM Interpretability Heatmap")
    ],
    title="Pneumonia Detector AI (Interpretability Enabled)",
    description="Upload a chest X-ray. The AI will predict pneumonia probability and generate a Grad-CAM heatmap highlighting the lung regions that drove the prediction."
)

interface.launch()