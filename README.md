# AgroIntel: Modern Farm Intelligence Platform 🌿

AgroIntel is an end-to-end AI-powered agricultural advisory platform designed to help farmers make data-driven decisions. By leveraging Machine Learning and Deep Learning, the application provides accurate recommendations for crops, fertilizers, and plant disease detection.

## 🚀 Features

### 1. Crop Recommendation 🌾
- **Algorithm**: Random Forest Classifier.
- **Input**: Soil parameters (Nitrogen, Phosphorous, Potassium, pH) and Climate parameters (Temperature, Humidity, Rainfall).
- **Output**: Predicts the most suitable crop to grow with confidence scores and growing tips.

### 2. Fertilizer Suggestion 🧪
- **Functionality**: Analyzes current soil nutrient levels against the requirements of a specific crop.
- **Output**: Recommends specific fertilizers to address deficiencies (N, P, or K).

### 3. Plant Disease Detection 🔍
- **Architecture**: Deep Learning using **ResNet9** (PyTorch).
- **Detection**: Supports 38 different classes of plant diseases across various crops (Tomato, Potato, Apple, Corn, etc.).
- **Input**: Upload an image of the plant leaf for instant diagnosis and treatment suggestions.

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **Machine Learning**: Scikit-learn, XGBoost
- **Deep Learning**: PyTorch, Torchvision
- **Data Handling**: Pandas, Numpy
- **Frontend**: HTML5, Vanilla CSS, Bootstrap 4, Lucide Icons
- **Deployment**: Gunicorn (Ready for Heroku/Cloud)

## 📋 Installation Guide

Follow these steps to set up the project on your local machine or a new device.

### Prerequisites
- Python 3.8+ installed.
- Git installed.

### Step-by-Step Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Bikramdash/AgroIntel.git
   cd AgroIntel
   ```

2. **Create a Virtual Environment**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   cd app
   python app.py
   ```
   The application will be available at `http://127.0.0.1:5000/`.

## 📂 Project Structure

```text
AgroIntel/
├── app/
│   ├── Data/               # CSV datasets for fertilizers
│   ├── models/             # Pre-trained ML & DL models (.pkl, .pth)
│   ├── static/             # CSS, Images, JS
│   ├── templates/          # HTML templates (Flask)
│   ├── utils/              # Helper scripts for disease & fertilizer logic
│   └── app.py              # Main Flask application
├── Data-raw/               # Raw datasets used for training
├── Data-processed/         # Processed datasets
├── notebooks/              # Jupyter notebooks for model training
├── train_crop_model.py     # Script to train the crop model
├── train_disease_model.py  # Script to train the disease model
├── requirements.txt        # Python dependencies
└── .gitignore              # Files to ignore in Git
```

## 🤝 Contributing
Contributions are welcome! If you have suggestions for new features or improvements, feel free to open an issue or submit a pull request.

## 📄 License
This project is licensed under the MIT License.

---
Developed with ❤️ to empower farmers through technology.
