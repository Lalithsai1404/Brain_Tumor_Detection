🧠 Brain Tumor Detection using Deep Learning
Python
TensorFlow
OpenCV
License
Status

📘 Overview
This project is a Deep Learning-based Brain Tumor Detection System that classifies MRI brain scans into four categories:

🧬 Glioma Tumor

🧠 Meningioma Tumor

🚫 No Tumor

🩺 Pituitary Tumor

The model uses Transfer Learning (MobileNetV2) from TensorFlow Keras to achieve high accuracy and efficient performance, even on limited datasets.

📂 Project Structure
Brain_Tumor_Detection/
│
├── data/
│ ├── Training/
│ │ ├── glioma_tumor/
│ │ ├── meningioma_tumor/
│ │ ├── no_tumor/
│ │ └── pituitary_tumor/
│ └── Testing/
│ ├── glioma_tumor/
│ ├── meningioma_tumor/
│ ├── no_tumor/
│ └── pituitary_tumor/
│
├── model/
│ └── brain_tumor_model.h5
│
├── train.py
├── predict.py
└── README.md

⚙️ Installation
Clone the repository
git clone https://github.com/<your-username>/Brain_Tumor_Detection.git
cd Brain_Tumor_Detection

Install dependencies
pip install -r requirements.txt

or manually:
pip install tensorflow opencv-python matplotlib numpy

Dataset Setup
Place your MRI dataset inside the data/ directory in the same structure shown above.

🚀 Usage
🧩 Train the Model
Train your model from scratch using:
python train.py

This will:

Load and augment the dataset

Train MobileNetV2 for classification

Save the best model as model/brain_tumor_model.h5

🔍 Run Predictions
Run the trained model on a single image:
python predict.py data/Testing/glioma_tumor/image.jpg

Sample Output:
Prediction: pituitary_tumor
Confidence: 0.924

🧠 Model Details
Parameter	Value
Base Model	MobileNetV2 (ImageNet Pretrained)
Input Size	224 x 224 x 3
Batch Size	32
Optimizer	Adam
Loss	Categorical Crossentropy
Metrics	Accuracy
Epochs	15
Classes	4 (glioma, meningioma, no_tumor, pituitary)

📊 Training Visualization
After training, accuracy graphs are automatically plotted using Matplotlib:
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.show()

🖼️ Sample Results
MRI Scan	Model Prediction
Glioma Tumor (Confidence: 0.92)

(Replace the above image URL with your own once uploaded.)

🧩 Key Features
✅ Transfer Learning with MobileNetV2
✅ Data Augmentation for improved generalization
✅ Real-time single image prediction
✅ Early Stopping and Model Checkpoint callbacks
✅ Lightweight and easy to deploy

🛠️ Technologies Used
Python 3.8+

TensorFlow / Keras

NumPy & Matplotlib

OpenCV for image preprocessing

📜 License
This project is licensed under the MIT License — see the LICENSE file for details.

🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.