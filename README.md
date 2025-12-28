# Plants Diseases Recognition System

A deep learning-based plant disease detection and classification system using Convolutional Neural Networks (CNN) built with TensorFlow and Keras. This project provides an interactive web interface powered by Streamlit for real-time plant disease diagnosis.

## 🌿 Project Overview

This system can identify and classify 38 different plant diseases across multiple crop types including:
- **Apple**: Apple scab, Black rot, Cedar apple rust, Healthy
- **Corn (Maize)**: Cercospora leaf spot, Common rust, Northern Leaf Blight, Healthy
- **Grape**: Black rot, Esca, Leaf blight, Healthy
- **Tomato**: Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites, Target Spot, Tomato mosaic virus, Yellow Leaf Curl Virus, Healthy
- **Potato**: Early blight, Late blight, Healthy
- **Pepper, Bell**: Bacterial spot, Healthy
- **And many more crops**: Cherry, Blueberry, Orange, Peach, Raspberry, Soybean, Squash, Strawberry

## 🎯 Features

- **Deep Learning Model**: Custom CNN architecture trained on PlantVillage dataset
- **Interactive Web Interface**: User-friendly Streamlit application
- **Real-time Predictions**: Upload plant images and get instant disease diagnosis
- **Confidence Scores**: View prediction confidence with visual indicators
- **Treatment Recommendations**: Detailed treatment and prevention strategies for each disease
- **Image Preprocessing**: Multiple image enhancement options (Original, Grayscale, Edge Detection, etc.)
- **Model Performance Visualization**: Interactive charts showing training metrics and class distribution
- **Batch Analysis**: Analyze multiple images at once

## 🛠️ Technologies Used

- **Deep Learning Framework**: TensorFlow, Keras
- **Web Framework**: Streamlit
- **Image Processing**: OpenCV, PIL
- **Data Handling**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Machine Learning**: scikit-learn

## 📋 Requirements

```
tensorflow
streamlit
numpy
pandas
pillow
opencv-python
matplotlib
seaborn
plotly
scikit-learn
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/salmankhan-1106/Plants-Diseases-Recognition-System.git
cd Plants-Diseases-Recognition-System
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Ensure you have the trained model file `best_model.h5` in the project directory

## 💻 Usage

### Running the Streamlit Application

```bash
streamlit run main.py
```

The application will open in your default web browser. You can then:
1. Upload a plant leaf image
2. Select preprocessing options if needed
3. Click "Classify Disease" to get predictions
4. View disease information, confidence scores, and treatment recommendations

### Training the Model

Open and run the Jupyter notebook:
```bash
jupyter notebook final.ipynb
```

The notebook contains:
- Data loading and preprocessing
- Model architecture definition
- Training pipeline with data augmentation
- Model evaluation and visualization
- Model saving functionality

## 📊 Model Architecture

The CNN model consists of:
- Multiple convolutional layers with batch normalization
- MaxPooling layers for dimensionality reduction
- Dropout layers for regularization
- Dense layers for classification
- Softmax activation for multi-class classification

**Input**: 224x224 RGB images
**Output**: 38 disease classes with confidence scores

## 📁 Project Structure

```
.
├── final.ipynb              # Training notebook
├── main.py                  # Streamlit application
├── requirements.txt         # Python dependencies
├── best_model.h5           # Trained model weights
├── notebook_overview.md    # Detailed notebook documentation
└── README.md               # Project documentation
```

## 📈 Model Performance

The model achieves high accuracy on the PlantVillage dataset with comprehensive evaluation metrics including:
- Training/Validation accuracy curves
- Confusion matrix
- Classification report with precision, recall, and F1-scores
- Per-class performance analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👥 Contributors

- [Salman Khan](https://github.com/salmankhan-1106)
- [Ayesha Mehmood](https://github.com/blackwatermelon0000)

## 🙏 Acknowledgments

- PlantVillage dataset for providing the training data
- TensorFlow and Keras teams for the deep learning framework
- Streamlit for the web application framework

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This is an educational project developed as part of a Machine Learning course. The model should be used as a supplementary tool and not as a replacement for professional agricultural advice.
