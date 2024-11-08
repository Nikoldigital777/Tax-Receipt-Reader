import os
import re
import shutil
import logging
import hashlib
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import easyocr
from PIL import Image, ImageEnhance
import pdf2image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pytesseract
from forex_python.converter import CurrencyRates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import boto3
from botocore.exceptions import ClientError
import yaml
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tax_processor.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class ReceiptData:
    """Data class for storing receipt information."""
    file_name: str
    date: str
    category: str
    amount: float
    currency: str
    vendor: str
    text: str
    confidence: float
    hash: str
    metadata: Dict
    
class TaxCategories:
    """Tax category management with YAML configuration."""
    def __init__(self, config_file: str = 'tax_categories.yaml'):
        self.config_file = config_file
        self.categories = self._load_categories()
        self.ml_classifier = self._initialize_classifier()
        
    def _load_categories(self) -> Dict:
        """Load tax categories from YAML configuration."""
        try:
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logging.warning(f"Config file {self.config_file} not found. Using default categories.")
            return self._get_default_categories()
            
    def _get_default_categories(self) -> Dict:
        """Provide default tax categories with keywords and rules."""
        return {
            'Business Expenses': {
                'keywords': ['office', 'software', 'hardware', 'professional'],
                'max_amount': 5000,
                'requires_approval': False
            },
            'Travel Expenses': {
                'keywords': ['airline', 'hotel', 'car rental', 'meals'],
                'max_amount': 10000,
                'requires_approval': True
            },
            # Add more categories with detailed rules
        }

    def _initialize_classifier(self) -> Tuple[MultinomialNB, TfidfVectorizer]:
        """Initialize ML classifier for categorization."""
        vectorizer = TfidfVectorizer(max_features=1000)
        classifier = MultinomialNB()
        return classifier, vectorizer

class ReceiptProcessor:
    """Enhanced receipt processing with ML capabilities."""
    def __init__(self):
        self.reader = easyocr.Reader(['en'])
        self.categories = TaxCategories()
        self.currency_converter = CurrencyRates()
        self.s3_client = self._initialize_s3_client()
        
    def _initialize_s3_client(self) -> Optional[boto3.client]:
        """Initialize AWS S3 client for cloud backup."""
        try:
            return boto3.client('s3')
        except Exception as e:
            logging.error(f"Failed to initialize S3 client: {e}")
            return None
            
    def preprocess_image(self, image: Image) -> Image:
        """Enhance image quality for better OCR results."""
        # Convert to grayscale
        img_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh)
        
        # Convert back to PIL Image
        return Image.fromarray(denoised)

    def extract_text(self, image_path: str) -> Tuple[str, float]:
        """Extract text from image with confidence score."""
        try:
            # Use both EasyOCR and Tesseract for better accuracy
            easy_result = self.reader.readtext(image_path)
            easy_text = ' '.join([text for _, text, conf in easy_result])
            easy_conf = np.mean([conf for _, _, conf in easy_result])
            
            tess_text = pytesseract.image_to_string(Image.open(image_path))
            
            # Combine results with weights based on confidence
            combined_text = f"{easy_text} {tess_text}"
            return combined_text, easy_conf
        except Exception as e:
            logging.error(f"Text extraction failed: {e}")
            return "", 0.0

    def extract_metadata(self, text: str, image_path: str) -> Dict:
        """Extract additional metadata from receipt."""
        metadata = {
            'image_size': os.path.getsize(image_path),
            'creation_date': datetime.fromtimestamp(
                os.path.getctime(image_path)
            ).isoformat(),
            'modification_date': datetime.fromtimestamp(
                os.path.getmtime(image_path)
            ).isoformat(),
            'extracted_emails': re.findall(r'[\w\.-]+@[\w\.-]+', text),
            'phone_numbers': re.findall(
                r'\+?[\d\s-]{10,}', text
            ),
            'tax_ids': re.findall(
                r'Tax ID:?\s*[\d-]+', text
            )
        }
        return metadata

    def detect_duplicate(self, file_path: str, text: str) -> Optional[str]:
        """Detect duplicate receipts using content hash."""
        content_hash = hashlib.md5(text.encode()).hexdigest()
        image_hash = hashlib.md5(
            Image.open(file_path).tobytes()
        ).hexdigest()
        return f"{content_hash}_{image_hash}"

    def validate_receipt(self, receipt: ReceiptData) -> List[str]:
        """Validate receipt data against business rules."""
        violations = []
        category_rules = self.categories.categories.get(receipt.category, {})
        
        if 'max_amount' in category_rules:
            if receipt.amount > category_rules['max_amount']:
                violations.append(
                    f"Amount exceeds maximum for category: {receipt.amount}"
                )
                
        if 'requires_approval' in category_rules:
            if category_rules['requires_approval'] and receipt.amount > 1000:
                violations.append("Receipt requires manager approval")
                
        # Add more validation rules
        return violations

    def backup_to_cloud(self, file_path: str, receipt: ReceiptData):
        """Backup receipt to cloud storage."""
        if not self.s3_client:
            return
            
        try:
            bucket_name = "tax-receipts-backup"
            key = f"{receipt.date}/{receipt.category}/{os.path.basename(file_path)}"
            self.s3_client.upload_file(file_path, bucket_name, key)
            logging.info(f"Backed up receipt to S3: {key}")
        except ClientError as e:
            logging.error(f"Failed to backup receipt: {e}")

class TaxReceiptGUI(tk.Tk):
    """Enhanced GUI with data visualization and advanced features."""
    def __init__(self):
        super().__init__()
        
        self.title("Advanced Tax Receipt Processor")
        self.geometry("1200x800")
        
        self.processor = ReceiptProcessor()
        self.setup_ui()
        
    def setup_ui(self):
        """Set up enhanced user interface."""
        # Create main container
        main_container = ttk.Frame(self)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook for different views
        notebook = ttk.Notebook(main_container)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Add tabs
        self.setup_processing_tab(notebook)
        self.setup_visualization_tab(notebook)
        self.setup_settings_tab(notebook)
        
    def setup_processing_tab(self, notebook):
        """Set up receipt processing interface."""
        process_frame = ttk.Frame(notebook)
        notebook.add(process_frame, text="Process Receipts")
        
        # Add processing controls
        ttk.Button(
            process_frame, 
            text="Select Receipts",
            command=self.process_receipts
        ).pack(pady=10)
        
        # Add progress bar
        self.progress = ttk.Progressbar(
            process_frame, 
            mode='determinate'
        )
        self.progress.pack(fill=tk.X, pady=10)
        
        # Add receipt list
        self.receipt_tree = ttk.Treeview(
            process_frame,
            columns=('Date', 'Category', 'Amount', 'Vendor'),
            show='headings'
        )
        self.receipt_tree.pack(fill=tk.BOTH, expand=True)
        
    def setup_visualization_tab(self, notebook):
        """Set up data visualization interface."""
        viz_frame = ttk.Frame(notebook)
        notebook.add(viz_frame, text="Visualizations")
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def setup_settings_tab(self, notebook):
        """Set up settings and configuration interface."""
        settings_frame = ttk.Frame(notebook)
        notebook.add(settings_frame, text="Settings")
        
        # Add settings controls
        ttk.Label(
            settings_frame,
            text="Tax Year:"
        ).pack(pady=5)
        
        self.tax_year = ttk.Entry(settings_frame)
        self.tax_year.pack(pady=5)
        
        ttk.Button(
            settings_frame,
            text="Save Settings",
            command=self.save_settings
        ).pack(pady=10)
        
    def process_receipts(self):
        """Handle receipt processing workflow."""
        files = filedialog.askopenfilenames(
            filetypes=[
                ("Images & PDFs", "*.jpg *.jpeg *.png *.pdf")
            ]
        )
        
        if not files:
            return
            
        self.progress['maximum'] = len(files)
        processed_data = []
        
        for i, file_path in enumerate(files):
            try:
                # Process receipt
                receipt_data = self.processor.process_receipt(file_path)
                
                # Validate receipt
                violations = self.processor.validate_receipt(receipt_data)
                if violations:
                    messagebox.showwarning(
                        "Validation Warnings",
                        "\n".join(violations)
                    )
                
                # Backup to cloud
                self.processor.backup_to_cloud(file_path, receipt_data)
                
                # Update UI
                self.receipt_tree.insert(
                    '',
                    'end',
                    values=(
                        receipt_data.date,
                        receipt_data.category,
                        f"{receipt_data.currency} {receipt_data.amount:.2f}",
                        receipt_data.vendor
                    )
                )
                
                processed_data.append(receipt_data)
                self.progress['value'] = i + 1
                self.update_idletasks()
                
            except Exception as e:
                logging.error(f"Failed to process receipt {file_path}: {e}")
                messagebox.showerror(
                    "Error",
                    f"Failed to process {os.path.basename(file_path)}"
                )
                
        self.update_visualizations(processed_data)
        
    def update_visualizations(self, data: List[ReceiptData]):
        """Update data visualizations."""
        self.ax.clear()
        
        # Create category summary
        categories = [d.category for d in data]
        amounts = [d.amount for d in data]
        
        # Plot bar chart
        self.ax.bar(categories, amounts)
        self.ax.set_title('Expenses by Category')
        self.ax.set_xlabel('Category')
        self.ax.set_ylabel('Amount')
        plt.xticks(rotation=45)
        
        self.fig.tight_layout()
        self.fig.canvas.draw()
        
    def save_settings(self):
        """Save application settings."""
        # Implementation for saving settings
        pass

def main():
    """Main application entry point."""
    app = TaxReceiptGUI()
    app.mainloop()

if __name__ == '__main__':
    main()
