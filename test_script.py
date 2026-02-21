import numpy as np
from PIL import Image
import os
import model_utils
import pdf_utils

print("Creating dummy image...")
img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
img = Image.fromarray(img_array)
img.save("dummy_xray.jpg")

print("Loading model...")
model = model_utils.get_model()

print("Running inference...")
label, prob, heatmap = model_utils.run_inference(model, "dummy_xray.jpg")
print(f"Label: {label}, Probability: {prob}, Heatmap path: {heatmap}")

print("Generating PDF...")
pdf_path = pdf_utils.generate_pdf_report("dummy_xray.jpg", heatmap, label, prob, "dummy_report.pdf")
print(f"PDF Generated at: {pdf_path}")
print("Test complete and successful.")
