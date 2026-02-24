🛡️ PneumoDetect AI: Privacy-First Medical Assistant
PneumoDetect AI is a high-performance, local-first medical imaging tool designed to detect Pneumonia from chest X-rays. It combines deep learning with Explainable AI (XAI) to show users exactly which regions of an image influenced the model's decision.

🚀 Key Features
Pneumonia Detection: Uses a fine-tuned MobileNetV2 architecture for high accuracy with low latency.

Grad-CAM Visualization: Generates heatmaps to highlight lung opacities, providing transparency for clinical decision support.

Privacy-First: Designed to run locally using Streamlit and Ollama, ensuring patient data never leaves the machine.

Interactive Chat: Integration with Llama 3.2 to provide contextual information about the findings.

🛠️ Tech Stack
Frontend: Streamlit

Deep Learning: TensorFlow / Keras (MobileNetV2)

Computer Vision: OpenCV, PIL

Explainability: Grad-CAM (Gradient-weighted Class Activation Mapping)

Local LLM: Ollama (Llama 3.2)

📦 Installation & Setup
1. Clone the Repository
Bash
git clone https://github.com/your-username/pneumodetect-ai.git
cd pneumodetect-ai
2. Set Up Virtual Environment
Bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
3. Model Preparation
Ensure your trained model file pneumodetect_model_final.keras is in the root directory.

Note: If the model file was ignored by git, ensure you remove it from .gitignore and push it to your repo.

4. Run the Application
Bash
streamlit run app.py
🔍 How Grad-CAM Works in this App
The assistant doesn't just give a "Yes/No" answer. It utilizes the gradients of the last convolutional layer in the MobileNetV2 backbone to produce a localization map.

Feature Extraction: The X-ray passes through the convolutional layers.

Gradient Calculation: The app calculates the gradient of the "Pneumonia" class score with respect to the feature maps.

Heatmap Generation: Positive gradients are pooled and overlaid on the original X-ray to show "hot zones" of infection.

⚠️ Disclaimer
This tool is for educational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider.
