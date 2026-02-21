import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import cv2
from PIL import Image

# Use tf.keras instead of keras to avoid compatibility issues
keras = tf.keras

def get_model():
    """
    Loads the fine-tuned Pneumonia detection model.
    """
    model_path = os.path.join(os.path.dirname(__file__), "pneumodetect_model_final.keras")
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}. Please ensure training is complete.")

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Loads an image from file, resizes it, and applies preprocessing
    required for MobileNetV2.
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    # Expanding dimensions to match batch size 1
    img_array = np.expand_dims(img_array, axis=0)
    # Preprocess input suitable for MobileNetV2
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

def get_last_conv_layer_name(model):
    """
    Finds the last convolutional layer in the model.
    """
    for layer in reversed(model.layers):
        try:
            if hasattr(layer, 'output') and layer.output is not None:
                if len(layer.output.shape) == 4:
                    return layer.name
        except Exception:
            pass
    raise ValueError("Could not find a 4D output layer.")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generates a Grad-CAM heatmap for a given image.
    """
    # Create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            # For a binary classifier with 1 output unit, preds has shape (1, 1)
            # We want to compute gradient w.r.t to the predicted probability.
            class_channel = preds[:, 0]
        else:
            class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron w.r.t. the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array by "how important this channel is" 
    last_conv_layer_output = last_conv_layer_output[0]
    
    # We must explicitly cast to float32 before dot product to avoid type errors
    pooled_grads = tf.cast(pooled_grads, dtype=tf.float32)
    last_conv_layer_output = tf.cast(last_conv_layer_output, dtype=tf.float32)
    
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    # applying ReLU to only keep features that have a positive influence on the target class
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(image_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    """
    Overlays the Grad-CAM heatmap on the original image and saves the result.
    """
    # Load the original image
    img = keras.preprocessing.image.load_img(image_path)
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Create an image with RGB colorized heatmap
    jet = keras.preprocessing.image.array_to_img(jet)
    jet = jet.resize((img.shape[1], img.shape[0]))
    jet = keras.preprocessing.image.img_to_array(jet)

    # Superimpose the heatmap on original image
    superimposed_img = jet * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)
    return cam_path

def run_inference(model, image_path, cam_path="cam.jpg"):
    """
    Runs inference on an image and generates a Grad-CAM heatmap.
    Returns: (prediction_label, probability_score, heatmap_path)
    """
    img_array = load_and_preprocess_image(image_path)
    
    # Run prediction
    preds = model.predict(img_array)
    probability = float(preds[0][0])
    
    # Binary threshold at 0.5
    label = "Pneumonia Detected" if probability > 0.5 else "Normal"
    
    # Generate Grad-CAM Heatmap
    last_conv_layer_name = get_last_conv_layer_name(model)
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    
    # Save superimposed image
    saved_cam_path = save_and_display_gradcam(image_path, heatmap, cam_path)
    
    return label, probability, saved_cam_path

