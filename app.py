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

# Custom CSS for dark modern aesthetics matching the reference
st.markdown("""
    <style>
    /* Main Background - Deep Purple/Dark */
    .stApp {
        background-color: #0b0f19; /* Dark blue/purple background */
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    h1, h2, h3, h4, h5, h6, p, span, div {
        color: #ffffff !important;
    }
    
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0px;
        letter-spacing: 1px;
    }
    
    .sub-title {
        text-align: center;
        font-size: 1rem;
        font-weight: 500;
        color: #9ca3af !important;
        margin-top: 5px;
        margin-bottom: 40px;
    }
    
    .metrics-container {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-bottom: 40px;
    }
    
    /* Stat Cards */
    .stat-card {
        background-color: #1f2937; /* Dark gray card */
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 15px 25px;
        text-align: center;
        min-width: 150px;
    }
    
    .stat-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #60a5fa !important; /* Light blue accent */
        margin-bottom: 5px;
    }
    
    .stat-label {
        font-size: 0.75rem;
        color: #9ca3af !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Result Cards */
    .result-card {
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        margin: 20px auto;
        max-width: 400px;
        animation: fadeIn 0.8s ease-in-out;
    }
    
    .result-Pneumonia {
        background-color: rgba(239, 68, 68, 0.1);
        border: 1px solid #ef4444;
    }
    
    .result-Pneumonia .result-title {
        color: #ef4444 !important;
    }
    
    .result-Normal {
        background-color: rgba(34, 197, 94, 0.1);
        border: 1px solid #22c55e;
    }
    
    .result-Normal .result-title {
        color: #22c55e !important;
    }

    .result-title {
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .result-conf {
        font-size: 18px;
        font-weight: 500;
        opacity: 0.9;
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 60px;
        padding-top: 20px;
        border-top: 1px solid #374151;
        color: #9ca3af !important;
        font-size: 0.85rem;
    }

    /* Hide Streamlit Header/Footer */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return model_utils.get_model()

def main():
    # Centered Header
    st.markdown("<div style='text-align: center; font-size: 48px; margin-bottom: 10px;'>🫁</div>", unsafe_allow_html=True)
    st.markdown("<h1 class='main-title'>PneumoDetect AI</h1>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Clinical-Grade AI for Rapid Pneumonia Detection<br><span style='font-size: 0.8rem; font-weight: normal;'>Fast • Accurate • Trusted</span></div>", unsafe_allow_html=True)
    
    # Custom Metrics Section
    col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 1])
    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">🎯 88.8%</div>
            <div class="stat-label">Model Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">⚡ < 2 sec</div>
            <div class="stat-label">Analysis Time</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">🔍 Grad-CAM</div>
            <div class="stat-label">Explainability</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col_upload_left, col_upload_main, col_upload_right = st.columns([1, 4, 1])
    
    with col_upload_main:
        st.markdown("<div style='text-align: center; margin-bottom: 10px;'>📥 <b>Upload & Analyze Chest X-Ray</b><br><span style='color:#9ca3af; font-size:0.8rem;'>Supports JPG, JPEG, PNG</span></div>", unsafe_allow_html=True)
        
        # Load model silently
        model = load_model()

        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

        if uploaded_file is not None:
            # Create a temporary directory to store files for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded image
                input_image_path = os.path.join(temp_dir, "input_image.jpg")
                with open(input_image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.markdown("<br>", unsafe_allow_html=True)
                col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
                with col_btn2:
                    run_btn = st.button("🚀 Analyze X-Ray Image", type="primary", use_container_width=True)
                
                if run_btn:
                    with st.spinner("Analyzing image and generating Grad-CAM heatmap..."):
                        # Process and run inference
                        heatmap_path = os.path.join(temp_dir, "heatmap.jpg")
                        try:
                            label, prob, heatmap_path = model_utils.run_inference(model, input_image_path, heatmap_path)
                            
                            # Display Results
                            st.markdown("---")
                            
                            # Display styled result
                            result_class = "result-Pneumonia" if "Pneumonia" in label else "result-Normal"
                            card_html = f"""
                            <div class="result-card {result_class}">
                                <div class="result-title">{label}</div>
                                <div class="result-conf">Confidence Score: {prob:.1%}</div>
                            </div>
                            """
                            st.markdown(card_html, unsafe_allow_html=True)
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            col_img1, col_img2 = st.columns(2)
                            
                            with col_img1:
                                st.markdown("<div style='text-align: center; color: #9ca3af;'>📷 Original X-Ray</div>", unsafe_allow_html=True)
                                st.image(input_image_path, width=None, use_container_width=True)
                                
                            with col_img2:
                                st.markdown("<div style='text-align: center; color: #9ca3af;'>🔥 Grad-CAM Heatmap</div>", unsafe_allow_html=True)
                                st.image(heatmap_path, width=None, use_container_width=True)
                                
                            st.markdown("<div style='text-align: center; font-size: 0.8rem; color: #6b7280; margin-top: 10px;'>🔴 Warmer colors (red/yellow) indicate areas that strongly influenced the model's prediction.</div>", unsafe_allow_html=True)
                            
                            # Generate PDF Report
                            report_path = os.path.join(temp_dir, "report.pdf")
                            pdf_utils.generate_pdf_report(input_image_path, heatmap_path, label, prob, report_path)
                            
                            with open(report_path, "rb") as pdf_file:
                                PDFbyte = pdf_file.read()

                            st.markdown("---")
                            col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
                            with col_dl2:
                                st.download_button(
                                    label="📄 Download Clinical PDF Report",
                                    data=PDFbyte,
                                    file_name="PneumoDetect_Report.pdf",
                                    mime="application/octet-stream",
                                    use_container_width=True
                                )
                        except Exception as e:
                            st.error(f"Error during analysis: {str(e)}")
                            
    # Footer Information Box
    st.markdown("""
    <div style='margin-top: 50px; padding: 20px; background-color: rgba(31, 41, 55, 0.5); border: 1px solid #374151; border-radius: 8px; text-align: center;'>
        <div style='color: #fbbf24; font-weight: bold; margin-bottom: 5px;'>⚠️ Medical Disclaimer</div>
        <div style='color: #9ca3af; font-size: 0.85rem;'>
            This tool is intended for research and educational purposes only.<br>
            Please seek advice from a qualified healthcare professional before making medical decisions.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='footer'>
        <p>Built with ❤️ using Streamlit, TensorFlow, and Python</p>
        <p>© 2026 PneumoDetect AI. All Rights Reserved.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
