"""
Medical Report Generation with PDF Export
Generates patient reports with analysis, test results, and annotated images
"""

import io
import os
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image as RLImage
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
import json


class PatientReport:
    """Generate and manage patient medical reports"""
    
    def __init__(self, patient_name, patient_id, age, gender, date=None):
        self.patient_name = patient_name
        self.patient_id = patient_id
        self.age = age
        self.gender = gender
        self.date = date or datetime.now()
        self.findings = []
        self.recommendations = []
        self.ultrasound_images = []
        self.segmentation_results = None
        self.disease_classification = None
        
    def add_ultrasound_finding(self, finding_text):
        """Add a finding from ultrasound examination"""
        self.findings.append({
            'timestamp': datetime.now(),
            'text': finding_text
        })
    
    def add_recommendation(self, recommendation_text):
        """Add a clinical recommendation"""
        self.recommendations.append({
            'timestamp': datetime.now(),
            'text': recommendation_text
        })
    
    def add_ultrasound_image(self, image_path, image_type="Original"):
        """Add ultrasound image to report"""
        self.ultrasound_images.append({
            'path': image_path,
            'type': image_type,
            'timestamp': datetime.now()
        })
    
    def set_segmentation_results(self, segmentation_data):
        """Set GCN segmentation results"""
        self.segmentation_results = segmentation_data
    
    def set_disease_classification(self, classification_data):
        """Set disease classification results"""
        self.disease_classification = classification_data
    
    def generate_summary(self):
        """Generate report summary"""
        summary = f"""
        PATIENT INFORMATION
        {'=' * 50}
        Name: {self.patient_name}
        Patient ID: {self.patient_id}
        Age: {self.age} years
        Gender: {self.gender}
        Examination Date: {self.date.strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        if self.disease_classification:
            summary += f"\n\nDISEASE CLASSIFICATION RESULTS\n{'=' * 50}\n"
            summary += f"Predicted Disease: {self.disease_classification['predicted_disease']}\n"
            summary += f"Confidence: {self.disease_classification['confidence']:.2f}%\n"
            
            if 'all_probabilities' in self.disease_classification:
                summary += "\nDetailed Probabilities:\n"
                for disease, prob in self.disease_classification['all_probabilities'].items():
                    summary += f"  - {disease}: {prob:.2f}%\n"
        
        if self.segmentation_results:
            summary += f"\n\nULTRASOUND SEGMENTATION RESULTS\n{'=' * 50}\n"
            summary += f"Infected Area Detected: Yes\n"
            summary += f"Infected Area Percentage: {self.segmentation_results['predicted_infected_area_percentage']:.2f}%\n"
        
        if self.findings:
            summary += f"\n\nFINDINGS\n{'=' * 50}\n"
            for i, finding in enumerate(self.findings, 1):
                summary += f"{i}. {finding['text']}\n"
        
        if self.recommendations:
            summary += f"\n\nRECOMMENDATIONS\n{'=' * 50}\n"
            for i, rec in enumerate(self.recommendations, 1):
                summary += f"{i}. {rec['text']}\n"
        
        return summary
    
    def create_annotated_image(self, original_image, segmentation_mask, output_path=None):
        """Create annotated image with infected area marked"""
        image = np.array(original_image).copy()
        
        # Create color overlay for infected areas
        mask_colored = np.zeros_like(image)
        mask_colored[segmentation_mask > 0] = [0, 255, 0]  # Green for infected areas
        
        # Overlay
        annotated = cv2.addWeighted(image, 0.7, mask_colored, 0.3, 0)
        
        # Add text annotations
        cv2.putText(annotated, "Infected Areas Marked in Green", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated, f"Date: {self.date.strftime('%Y-%m-%d')}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        
        return annotated


class PDFReportGenerator:
    """Generate PDF reports with images and patient details"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._customize_styles()
    
    def _customize_styles(self):
        """Customize paragraph styles"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f4788'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2c5aa0'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='NormalText',
            parent=self.styles['Normal'],
            fontSize=11,
            alignment=TA_JUSTIFY,
            spaceAfter=10
        ))
    
    def generate_pdf(self, patient_report, output_path, include_images=True):
        """
        Generate comprehensive PDF report
        """
        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=A4,
                               rightMargin=0.75*inch, leftMargin=0.75*inch,
                               topMargin=0.75*inch, bottomMargin=0.75*inch)
        
        story = []
        
        # Title
        title = Paragraph("MEDICAL DIAGNOSTIC REPORT", self.styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 0.3*inch))
        
        # Patient Information Table
        patient_data = [
            ['PATIENT INFORMATION', ''],
            ['Name', patient_report.patient_name],
            ['Patient ID', patient_report.patient_id],
            ['Age', f"{patient_report.age} years"],
            ['Gender', patient_report.gender],
            ['Examination Date', patient_report.date.strftime('%Y-%m-%d %H:%M:%S')]
        ]
        
        patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#2c5aa0')),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        story.append(patient_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Disease Classification Results
        if patient_report.disease_classification:
            story.append(Paragraph("DISEASE CLASSIFICATION RESULTS", self.styles['SectionHeading']))
            
            class_data = [
                ['Predicted Disease', patient_report.disease_classification['predicted_disease']],
                ['Confidence Level', f"{patient_report.disease_classification['confidence']:.2f}%"],
            ]
            
            if 'all_probabilities' in patient_report.disease_classification:
                class_data.append(['Probability Distribution', ''])
                for disease, prob in patient_report.disease_classification['all_probabilities'].items():
                    class_data.append(['', f"{disease}: {prob:.2f}%"])
            
            class_table = Table(class_data, colWidths=[2*inch, 4*inch])
            class_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#4472c4')),
                ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (1, 0), 11),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.lightblue, colors.white])
            ]))
            story.append(class_table)
            story.append(Spacer(1, 0.2*inch))
        
        # Segmentation Results
        if patient_report.segmentation_results:
            story.append(Paragraph("ULTRASOUND SEGMENTATION ANALYSIS", self.styles['SectionHeading']))
            
            seg_data = [
                ['Infected Area Detected', 'Yes'],
                ['Infected Area Percentage', f"{patient_report.segmentation_results['predicted_infected_area_percentage']:.2f}%"],
            ]
            
            seg_table = Table(seg_data, colWidths=[2*inch, 4*inch])
            seg_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#70ad47')),
                ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.lightgreen, colors.white])
            ]))
            story.append(seg_table)
            story.append(Spacer(1, 0.2*inch))
        
        # Findings
        if patient_report.findings:
            story.append(Paragraph("FINDINGS", self.styles['SectionHeading']))
            for finding in patient_report.findings:
                story.append(Paragraph(f"• {finding['text']}", self.styles['NormalText']))
            story.append(Spacer(1, 0.2*inch))
        
        # Recommendations
        if patient_report.recommendations:
            story.append(Paragraph("RECOMMENDATIONS", self.styles['SectionHeading']))
            for rec in patient_report.recommendations:
                story.append(Paragraph(f"• {rec['text']}", self.styles['NormalText']))
            story.append(Spacer(1, 0.2*inch))
        
        # Add images if available
        if include_images and patient_report.ultrasound_images:
            for img_data in patient_report.ultrasound_images:
                if os.path.exists(img_data['path']):
                    story.append(PageBreak())
                    story.append(Paragraph(f"ULTRASOUND IMAGE - {img_data['type']}", self.styles['SectionHeading']))
                    
                    # Add image
                    try:
                        img = RLImage(img_data['path'], width=6*inch, height=5*inch)
                        story.append(img)
                    except:
                        pass
                    
                    story.append(Spacer(1, 0.2*inch))
        
        # Footer
        story.append(Spacer(1, 0.3*inch))
        footer_text = "This report is confidential and intended only for the use of the patient and healthcare provider."
        story.append(Paragraph(footer_text, self.styles['Normal']))
        
        # Build PDF
        doc.build(story)
        return output_path


def generate_report_with_download(patient_name, patient_id, age, gender, 
                                 original_image_path, segmentation_results,
                                 disease_classification, findings, recommendations):
    """
    Convenience function to generate complete report
    Returns path to generated PDF
    """
    # Create patient report
    report = PatientReport(patient_name, patient_id, age, gender)
    
    # Add data
    report.add_ultrasound_image(original_image_path, "Original Ultrasound")
    report.set_segmentation_results(segmentation_results)
    report.set_disease_classification(disease_classification)
    
    for finding in findings:
        report.add_ultrasound_finding(finding)
    
    for rec in recommendations:
        report.add_recommendation(rec)
    
    # Generate PDF
    pdf_generator = PDFReportGenerator()
    output_file = f"patient_report_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf_path = pdf_generator.generate_pdf(report, output_file)
    
    return pdf_path
