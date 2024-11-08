# Local Receipt Processor

A privacy-focused, offline-first receipt processing solution that helps organize and categorize receipts for tax purposes without sending sensitive data to cloud services.

## 🔒 Privacy First

Built for those who want to digitize and organize receipts while keeping their financial data private. This tool runs completely offline on your local machine, using:
- EasyOCR - An offline, open-source OCR engine
- Tesseract - Google's open-source OCR engine that runs locally
- Local machine learning for improved categorization
- No cloud dependencies or data transmission

## ✨ Features

- **Offline Receipt Processing**
  - Image and PDF receipt support
  - Automatic text extraction
  - Multi-page document handling
  - Image preprocessing for better accuracy

- **Smart Categorization**
  - Automatic merchant detection
  - ML-powered category suggestions
  - Customizable category rules
  - Business expense validation

- **Financial Management**
  - Multi-currency support
  - Tax category mapping
  - Duplicate receipt detection
  - Expense rule validation

- **Data Organization**
  - Structured receipt archiving
  - Searchable receipt database
  - Export to CSV for spreadsheet analysis
  - Tax summary reports

- **Modern Interface**
  - Drag-and-drop receipt import
  - Real-time processing feedback
  - Interactive data visualizations
  - Category management UI

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/local-receipt-processor.git
cd local-receipt-processor
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Tesseract OCR:
- **Windows**: Download and install from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
- **MacOS**: `brew install tesseract`
- **Linux**: `sudo apt-get install tesseract-ocr`

## 📊 Usage

1. Launch the application:
```bash
python tax_processor.py
```

2. Select a folder containing your receipts

3. The application will:
   - Process each receipt
   - Extract relevant information
   - Categorize expenses
   - Generate reports

4. Review and adjust categorizations as needed

5. Export data for tax preparation

## 📁 Supported File Types

- Images: JPG, JPEG, PNG
- Documents: PDF (will be converted to images for processing)

## ⚙️ Configuration

Customize `tax_categories.yaml` to define:
- Business expense categories
- Category-specific rules
- Amount thresholds
- Required approvals

Example configuration:
```yaml
Business Expenses:
  keywords:
    - office
    - software
    - hardware
  max_amount: 5000
  requires_approval: false

Travel Expenses:
  keywords:
    - airline
    - hotel
    - meal
  max_amount: 10000
  requires_approval: true
```

## 📈 Data Visualization

The tool includes built-in visualizations for:
- Expense distribution by category
- Monthly spending trends
- Category-wise comparisons
- Tax deduction summaries

## 🛡️ Privacy Features

1. **Offline Processing**
   - All OCR and ML processing happens locally
   - No internet connection required
   - No data leaves your machine

2. **Data Security**
   - Local data storage only
   - No cloud backups unless explicitly configured
   - Optional file encryption

3. **Configurable Data Retention**
   - Control how long receipt data is stored
   - Secure data deletion options
   - Export and backup controls

## 🔧 Technical Details

Built with:
- Python 3.8+
- EasyOCR for primary text extraction
- Tesseract for secondary validation
- scikit-learn for ML categorization
- OpenCV for image preprocessing
- Tkinter for the user interface
- Pandas for data management
- Matplotlib for visualizations

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This tool is provided as-is for personal and small business use. While it aims to assist with tax preparation, please consult with a qualified tax professional for official tax advice and verification.

## 🙏 Acknowledgments

- EasyOCR team for the offline OCR engine
- Tesseract OCR project
- scikit-learn community
- All open-source contributors

---

Made with ❤️ for privacy-conscious individuals
