import streamlit as st
import os
from PIL import Image
import tempfile
import model_utils
import pdf_utils

# Set page configuration
st.set_page_config(
    page_title="PneumoDetect AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        background-color: #f8f9fa;
    }
    .css-1d391kg {
        background-color: #e9ecef;
    }
    h1 {
        color: #2c3e50;
    }
    .result-Pneumonia {
        color: #e74c3c;
        font-weight: bold;
        font-size: 24px;
        padding: 10px;
        border: 2px solid #e74c3c;
        border-radius: 5px;
        background-color: #fadbd8;
        text-align: center;
    }
    .result-Normal {
        color: #27ae60;
        font-weight: bold;
        font-size: 24px;
        padding: 10px;
        border: 2px solid #27ae60;
        border-radius: 5px;
        background-color: #d5f5e3;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return model_utils.get_model()

def main():
    st.title("🫁 PneumoDetect AI")
    st.markdown("### Explainable AI for Chest X-Ray Analysis")
    st.markdown("Upload a Chest X-ray image to detect signs of Pneumonia. The AI will provide a Grad-CAM heatmap highlighting the specific areas influencing its decision.")

    # Load model
    with st.spinner("Loading Fine-Tuned AI Model..."):
        model = load_model()

    st.sidebar.header("About PneumoDetect AI")
    st.sidebar.info("""
    This application uses a Convolutional Neural Network (MobileNetV2) fine-tuned on a chest X-ray dataset to detect Pneumonia.
    
    **How it works:**
    1. Upload a regular Chest X-ray.
    2. The AI analyzes the image.
    3. Review the Grad-CAM heatmap which highlights the focus areas.
    4. Download a PDF report.
    """)

    uploaded_file = st.file_uploader("Drag and drop a Chest X-Ray image here", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Create a temporary directory to store files for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded image
            input_image_path = os.path.join(temp_dir, "input_image.jpg")
            with open(input_image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.markdown("---")
            run_btn = st.button("Analyze Image", type="primary", use_container_width=True)
            
            if run_btn:
                with st.spinner("Analyzing image and generating Grad-CAM heatmap..."):
                    # Process and run inference
                    heatmap_path = os.path.join(temp_dir, "heatmap.jpg")
                    try:
                        label, prob, heatmap_path = model_utils.run_inference(model, input_image_path, heatmap_path)
                        
                        # Display Results
                        st.markdown("## AI Analysis Results")
                        
                        # Display styled result
                        result_class = "result-Pneumonia" if "Pneumonia" in label else "result-Normal"
                        st.markdown(f'<div class="{result_class}">{label} ({prob:.1%} Confidence)</div>', unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Original X-Ray")
                            st.image(input_image_path, use_container_width=True)
                            
                        with col2:
                            st.subheader("Grad-CAM Heatmap")
                            st.image(heatmap_path, use_container_width=True)
                            st.caption("Warmer colors (red/yellow) indicate areas that strongly influenced the model's prediction.")
                        
                        # Generate PDF Report
                        report_path = os.path.join(temp_dir, "report.pdf")
                        pdf_utils.generate_pdf_report(input_image_path, heatmap_path, label, prob, report_path)
                        
                        with open(report_path, "rb") as pdf_file:
                            PDFbyte = pdf_file.read()

                        st.markdown("---")
                        st.download_button(
                            label="📄 Download Clinical PDF Report",
                            data=PDFbyte,
                            file_name="PneumoDetect_Report.pdf",
                            mime="application/octet-stream",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()
