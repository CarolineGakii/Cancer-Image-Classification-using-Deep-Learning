# Cancer-Image-Classification-using-Deep-Learning
This project explores and compares the performance of three convolutional neural network architectures — ResNet50V2, DenseNet121, and CoAtNet — for classifying benign vs. malignant gastric histopathology images.  📌 Focus: Applying deep learning to locally acquired cancer images for accurate, explainable diagnosis and model evaluation.
📂 Project Overview
•	Problem: Classify gastric histology images as either benign or malignant
•	Dataset: Locally collected, organized into benign/ and malignant/ folders
•	Task: Binary image classification
•	Models Trained: ResNet50V2, DenseNet121, CoAtNet
•	Metrics Used: Accuracy, AUC, F1-score, Sensitivity, Specificity
•	Tools: TensorFlow/Keras, scikit-learn, matplotlib
________________________________________
🔍 Objectives
•	Evaluate deep learning models on local histopathological images
•	Compare traditional CNNs vs. hybrid architectures (CNN + Transformer)
•	Identify the best-performing model in terms of medical relevance (e.g., high sensitivity for malignant cases)
•	Visualize training trends and confusion matrices
•	Lay foundation for future deployment or Grad-CAM explainability
________________________________________
🖼️ Dataset
•	✅ Public gastric cancer histology images
•	🤁 Preprocessed to 224x224 resolution
•	📁 Directory structure:
 	organized_gastric/
  ├── benign/
  └── malignant/
•	✅ Split:
o	Training: 80%
o	Validation: 20%
________________________________________
🧠 Models Trained
ResNet50V2
CoAtNet
DenseNet121

📌 Observation: While CoAtNet achieved the highest malignant recall (sensitivity), ResNet50V2 delivered the most balanced performance across all metrics.
________________________________________
📈 Results & Visualizations
•	📉 Training and Validation Curves
ResNet and CoAtNet both showed smooth convergence. DenseNet underperformed in generalization.
•	🗞 Confusion Matrices Located under /visualizations/, they give insight into misclassification rates.
•	🔬 Classification Reports Detailed breakdowns of precision, recall, and F1 for both classes.
	Pred: Benign	Pred: Malignant
True Benign	620	152
True Malignant	76	706
________________________________________
🧪 Training Strategy
•	Transfer Learning with ImageNet weights
•	Fine-tuning of top layers
•	Data augmentation via ImageDataGenerator
•	Callbacks:
o	ModelCheckpoint for best model saving
o	EarlyStopping to prevent overfitting
________________________________________
📦 Installation
1.	Clone the repo:
git clone [https://github.com/yourusername/gastric-cancer-classification](https://github.com/CarolineGakii/Cancer-Image-Classification-using-Deep-Learning.git
cd gastric-cancer-classification
2.	Create a virtual environment and activate it:
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
3.	Install required libraries:
pip install -r requirements.txt
________________________________________
🚀 Usage
🔹 Training
python train_resnet.py  # Or train_coatnet.py, train_densenet.py
🔹 Evaluation
python evaluate_model.py --model models/resnet50v2_best.keras
🔹 Visualization
Plots and confusion matrices will be saved to:
/visualizations/
________________________________________
🧠 Insights & Discussion
•	ResNet50V2 outperformed CoAtNet despite the latter’s transformer hybrid architecture.
•	DenseNet121 underperformed in this context, possibly due to overfitting or incompatibility with the domain-specific texture features.
•	Locally collected images may benefit from further augmentation, contrast enhancement, or domain adaptation.
•	Future work could explore:
o	EfficientNetV2
o	Vision Transformers (ViT)
o	Grad-CAM for explainability
o	Web deployment (Flask/Streamlit)
________________________________________
📁 Folder Structure
🗁 gastric-cancer-classification/
├── models/
├── visualizations/
│   └── *.png                # Curves & matrices
├── requirements.txt
Result log
└── README.md
Author
Caroline Gakii
MSc. Data Science | Cancer AI Research | Conference speaker
