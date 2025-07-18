# 🛡️ Network Intrusion Detection

## 📌 Project Title:
**Network Intrusion Detection**

## 👨‍💻 Authors:
- Amar Hasanović  
- Sanjin Šabanović  

## 🎓 Faculty & Course:
University of Sarajevo  
Faculty of Electrical Engineering  
Course: Artificial Intelligence  
Supervisor: Amina Bajrić  
Date: June 2025

---

## 📖 Project Description

This project presents a machine learning approach for detecting network intrusions using the **Random Forest algorithm**. The system is trained and evaluated on the **NSL-KDD** dataset, a well-known benchmark dataset for intrusion detection. The main goal is to classify network connections as either **normal** or **anomalous** (attack), based on extracted features.

---

## 🧠 Objectives

- Load and preprocess the NSL-KDD dataset.
- Train a Random Forest classifier to detect anomalies.
- Evaluate model performance using test data.
- Visualize the results and analyze feature importance.

---

## 📁 Dataset

We use the **NSL-KDD** dataset from Kaggle, which contains labeled network traffic instances:
- [`train_data.csv`](./data/train_data.csv): used for training the model.
- [`test_data.csv`](./data/test_data.csv): used for testing the trained model.

**Dataset source:**  
[Kaggle – NSL-KDD by sampadab17](https://www.kaggle.com/datasets/sampadab17/nslkdd)

---

## ⚙️ Technologies Used

- **Python**
- **Scikit-learn**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**

---

## 🧪 Project Workflow

1. **Data Preprocessing**
   - Removed NaN values
   - Label encoding for categorical features
   - Feature standardization

2. **Model Training**
   - Trained a Random Forest classifier with tuned parameters
   - Evaluated using accuracy, precision, recall, F1-score

3. **Model Evaluation**
   - Confusion matrix visualization
   - ROC curve and accuracy graph
   - Feature importance ranking

---

## 📊 Results

- **Accuracy**: ~99% on training data  
- **Test performance**: Very high precision and recall on most classes  
- **Feature importance**: Top 15 features visualized and analyzed  
- The model successfully differentiates between normal and attack connections.

![Confusion Matrix](./images/confusion_matrix.png)

---

## 📂 Repository Structure
random-forest-nid/
│
├── data/
│ ├── train_data.csv
│ └── test_data.csv
│
├── notebooks/
│ └── random_forest_nid.ipynb
│
├── images/
│ └── confusion_matrix.png
│ └── accuracy_graph.png
│
├── requirements.txt
├── .gitignore
└── README.md

---

## ✅ Installation & Run Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/random-forest-nid.git
   cd random-forest-nid
   
2. Install dependencies:
pip install -r requirements.txt

3. Open the notebook:
jupyter notebook notebooks/random_forest_nid.ipynb

💡 Future Improvements
  Add deep learning models (e.g., neural networks) for comparison

  Explore unsupervised techniques (e.g., clustering)

  Deploy the model in real-time intrusion detection systems

  Automate preprocessing and model evaluation in a pipeline
  
