# Dzongkha Digit Classification Using Deep Learning Architectures

A comprehensive deep learning solution for recognizing traditional Dzongkha numerals (༠-༩) using computer vision techniques. This project implements and compares three distinct neural network architectures: Custom CNN, ResNet, and VGG16.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preparation](#data-preparation)
- [Model Architectures](#model-architectures)
- [Performance Metrics](#performance-metrics)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

## 🎯 Project Overview

### Objectives

This project addresses the critical challenge of digitizing Bhutanese cultural heritage by developing an automated recognition system for traditional Dzongkha numerals. The main objectives include:

- **Cultural Preservation**: Enable digital archival of historical documents containing Dzongkha numerals
- **Educational Applications**: Support learning tools for Dzongkha language education
- **Administrative Efficiency**: Automate data entry for government and institutional documents
- **Accessibility**: Improve digital accessibility for Dzongkha speakers and learners

### Problem Statement

Traditional OCR systems primarily focus on Latin scripts and Arabic numerals, with limited support for Dzongkha numerals (༠, ༡, ༢, ༣, ༤, ༥, ༦, ༧, ༨, ༩). This project fills that gap by providing a specialized deep learning solution.

## ✨ Features

- **Multi-Architecture Comparison**: Implementation of CNN, ResNet, and VGG16 architectures
- **Advanced Preprocessing**: Automated grid analysis and digit extraction from 17×10 grid images
- **High Accuracy**: Best model achieves 95.2% accuracy on test dataset
- **Production Ready**: Deployed on HuggingFace Spaces with user-friendly interface
- **Comprehensive Evaluation**: Detailed performance metrics and comparative analysis
- **Mixed Precision Training**: Optimized training with 40% speed improvement

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster training)
- 8GB+ RAM recommended

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/dzongkha-digit-classification.git
   cd dzongkha-digit-classification
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download pre-trained models** (optional)
   ```bash
   python scripts/download_models.py
   ```

## 📖 Usage

### Quick Start

1. **Single Image Prediction**
   ```python
   from src.models.predictor import DzongkhaDigitPredictor
   
   # Initialize predictor with best model (ResNet)
   predictor = DzongkhaDigitPredictor(model_path='models/resnet_best.pth')
   
   # Predict single digit
   result = predictor.predict('path/to/digit_image.png')
   print(f"Predicted digit: {result['digit']}")
   print(f"Confidence: {result['confidence']:.2f}")
   ```

2. **Batch Processing**
   ```python
   from src.data.batch_processor import BatchProcessor
   
   processor = BatchProcessor(model_path='models/resnet_best.pth')
   results = processor.process_directory('data/test_images/')
   ```

3. **Web Interface**
   ```bash
   python app.py
   # Open http://localhost:5000 in your browser
   ```

### Expected Outputs

**Single Prediction Output:**
```json
{
  "predicted_digit": "༣",
  "confidence": 0.96,
  "all_probabilities": {
    "༠": 0.01, "༡": 0.02, "༢": 0.01, "༣": 0.96,
    "༤": 0.00, "༥": 0.00, "༦": 0.00, "༧": 0.00,
    "༨": 0.00, "༩": 0.00
  }
}
```

**Training Output:**
```
Epoch 1/30: Train Loss: 2.1234, Train Acc: 45.2%, Val Loss: 1.8765, Val Acc: 52.1%
Epoch 5/30: Train Loss: 0.8234, Train Acc: 78.5%, Val Loss: 0.7123, Val Acc: 81.2%
...
Best Model: ResNet at Epoch 27 with 95.2% validation accuracy
```

## 📊 Data Preparation

### Dataset Structure

```
data/
├── raw/
│   ├── grid_images/          # Original 17×10 grid images
│   └── metadata.json         # Image metadata
├── processed/
│   ├── train/                # Training samples (70%)
│   ├── val/                  # Validation samples (15%)
│   └── test/                 # Test samples (15%)
└── extracted/
    └── digits/               # Individual digit images (28×28)
```

### Data Preparation Guidelines

1. **Grid Image Requirements**
   - Format: PNG or JPG
   - Resolution: Minimum 1000×600 pixels
   - Layout: 17 rows × 10 columns of digits
   - Clear digit boundaries with minimal noise

2. **Automated Extraction Process**
   ```bash
   python scripts/extract_digits.py --input data/raw/grid_images/ --output data/extracted/
   ```

3. **Quality Assessment**
   - Automatic quality scoring based on variance, edge content, and contrast
   - Minimum quality threshold: 0.3
   - Manual validation recommended for edge cases

4. **Data Augmentation**
   - Rotation: ±15 degrees
   - Translation: ±10%
   - Scaling: 0.9-1.1x
   - Applied during training only

### Dataset Statistics

- **Total Samples**: 3,264 high-quality digits
- **Distribution**: Balanced across all 10 digit classes
- **Quality Distribution**: 45% high, 35% medium, 20% low quality
- **Train/Val/Test Split**: 70%/15%/15%

## 🏗️ Model Architectures

### 1. Custom CNN
- **Architecture**: 3 convolutional blocks + 2 fully connected layers
- **Parameters**: 1.2M
- **Features**: Batch normalization, dropout regularization
- **Performance**: 94.8% test accuracy

### 2. Custom ResNet (Best Model)
- **Architecture**: ResNet blocks with skip connections
- **Parameters**: 890K (most efficient)
- **Features**: Adaptive average pooling, residual connections
- **Performance**: 95.2% test accuracy

### 3. Custom VGG16
- **Architecture**: VGG-inspired with multiple 3×3 convolutions
- **Parameters**: 2.1M
- **Features**: Deep feature extraction, adaptive pooling
- **Performance**: 93.6% test accuracy

### Training Configuration

```python
HYPERPARAMS = {
    'ResNet': {
        'learning_rate': 0.001,
        'max_lr': 0.005,
        'weight_decay': 1e-4,
        'dropout_rate': 0.3,
        'batch_size': 128,
        'epochs': 30
    }
}
```

## 📈 Performance Metrics

### Overall Results

| Model  | Test Accuracy | Training Epochs | Parameters | Early Stopped |
|--------|---------------|-----------------|------------|---------------|
| CNN    | 94.8%         | 23/30          | 1.2M       | Yes           |
| ResNet | **95.2%**     | 27/30          | 890K       | Yes           |
| VGG16  | 93.6%         | 30/30          | 2.1M       | No            |

### Per-Class Accuracy (ResNet - Best Model)

| Digit | Dzongkha | Accuracy |
|-------|----------|----------|
| 0     | ༠        | 96.8%    |
| 1     | ༡        | 97.1%    |
| 2     | ༢        | 95.9%    |
| 3     | ༣        | 94.7%    |
| 4     | ༤        | 93.9%    |
| 5     | ༥        | 94.2%    |
| 6     | ༦        | 93.8%    |
| 7     | ༧        | 95.5%    |
| 8     | ༨        | 96.1%    |
| 9     | ༩        | 94.8%    |

### Training Optimizations

- **OneCycleLR Scheduler**: Faster convergence
- **Mixed Precision Training**: 40% speed improvement
- **Early Stopping**: Prevents overfitting (patience=7)
- **Data Augmentation**: Improves generalization

## 🚀 Deployment

### HuggingFace Spaces

The model is deployed on HuggingFace Spaces for public access:
- **URL**: [https://huggingface.co/spaces/zsonam/dzongkha_digit_classification](https://huggingface.co/spaces/yourusername/dzongkha-digits)
- **Interface**: Gradio-based web interface
- **Features**: Image upload, webcam capture, drawing canvas


## 📁 Project Structure

```
dzongkha-digit-classification/
├── src/
│   ├── data/
│   │   ├── data_loader.py
│   │   ├── preprocessor.py
│   │   └── grid_extractor.py
│   ├── models/
│   │   ├── cnn.py
│   │   ├── resnet.py
│   │   ├── vgg16.py
│   │   └── predictor.py
│   ├── training/
│   │   ├── trainer.py
│   │   └── utils.py
│   └── evaluation/
│       ├── evaluator.py
│       └── metrics.py
├── scripts/
│   ├── extract_digits.py
│   ├── train_models.py
│   └── evaluate_models.py
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── model_comparison.ipynb
│   └── results_analysis.ipynb
├── models/
│   ├── resnet_best.pth
│   ├── cnn_best.pth
│   └── vgg16_best.pth
├── data/
├── app.py
├── requirements.txt
├── README.md
└── report.pdf
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

For detailed methodology, results analysis, and technical implementation details, please refer to the comprehensive project report: **[Dzongkha Digit Classification Report](DAM101_Report(02240365).pdf)**

### Key Publications

1. LeCun, Y., et al. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*.
2. He, K., et al. (2016). Deep residual learning for image recognition. *CVPR*.
3. Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. *arXiv*.

## 🙏 Acknowledgments

- Bhutanese cultural heritage preservation initiatives
- PyTorch and HuggingFace communities
- Google Colab for computational resources
- Contributors to open-source deep learning frameworks

