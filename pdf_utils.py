from fpdf import FPDF
from datetime import datetime
import matplotlib.pyplot as plt
import os
import tempfile

class ClinicalReportPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'PneumoDetect AI - Clinical Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(image_path, heatmap_path, prediction, probability, output_path="report.pdf"):
    pdf = ClinicalReportPDF()
    pdf.add_page()
    
    # Date and Time
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
    
    # Result Summary
    pdf.set_font('Arial', 'B', 16)
    if prediction == "Pneumonia Detected":
        pdf.set_text_color(220, 53, 69) # Red
    else:
        pdf.set_text_color(40, 167, 69) # Green
        
    pdf.cell(0, 10, f"AI Diagnosis: {prediction}", 0, 1)
    pdf.set_text_color(0, 0, 0) # Black
    
    pdf.set_font('Arial', '', 12)
    # Give detailed breakdown
    pneum_prob = probability if prediction == "Pneumonia Detected" else (1 - probability)
    normal_prob = 1 - pneum_prob
    
    pdf.cell(0, 8, f"Probability of Pneumonia: {pneum_prob:.2%}", 0, 1)
    pdf.cell(0, 8, f"Probability of Normal (Healthy): {normal_prob:.2%}", 0, 1)
    
    pdf.ln(5)
    
    # Generate a temporary Matplotlib Bar Chart
    with tempfile.TemporaryDirectory() as temp_dir:
        chart_path = os.path.join(temp_dir, 'confidence_chart.png')
        
        plt.figure(figsize=(6, 2))
        categories = ['Normal', 'Pneumonia']
        probs = [normal_prob * 100, pneum_prob * 100]
        colors = ['#28a745' if normal_prob > pneum_prob else '#6c757d', 
                  '#dc3545' if pneum_prob > normal_prob else '#6c757d']
        
        bars = plt.barh(categories, probs, color=colors)
        plt.xlim(0, 100)
        plt.xlabel('Confidence Percentage (%)')
        plt.title('AI Decision Breakdown')
        
        # Add text labels on bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 1, bar.get_y() + bar.get_height()/2., 
                     f'{width:.1f}%', ha='left', va='center')
            
        plt.tight_layout()
        plt.savefig(chart_path, dpi=150)
        plt.close()

        # Insert Chart Image
        curr_y = pdf.get_y()
        pdf.image(chart_path, x=10, y=curr_y, w=120)
        pdf.ln(55) # Move past chart

    # Images Section
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Visual Evidence Analysis', 0, 1)
    
    # Side by side images (X: 10, Y: curr_y)
    curr_y = pdf.get_y() + 5
    
    # Original Image
    pdf.set_font('Arial', 'I', 10)
    pdf.text(30, curr_y - 2, "Original X-Ray")
    # Image (x, y, w, h)
    pdf.image(image_path, x=10, y=curr_y, w=85)
    
    # Heatmap Image
    pdf.text(125, curr_y - 2, "Grad-CAM Heatmap (AI Focus Area)")
    pdf.image(heatmap_path, x=105, y=curr_y, w=85)
    
    # Move to bottom to add notes
    pdf.set_y(curr_y + 90)
    pdf.set_font('Arial', 'I', 10)
    pdf.multi_cell(0, 5, "Note: This is an AI-generated analysis based on a fine-tuned MobileNetV2 architecture. The Grad-CAM heatmap highlights the region most indicative of the prediction. The confidence chart indicates the exact probability distribution between the 'Normal' and 'Pneumonia' classifications across the analyzed X-ray features.\n\nDisclaimer: This report must be reviewed by a qualified radiologist or physician.")
    
    pdf.output(output_path)
    return output_path
