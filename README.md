<!-- Project Banner -->
<p align="center">
  <img src="https://img.shields.io/badge/Thyroid%20Disease%20Detection-ML-blueviolet?style=for-the-badge" alt="Project Banner"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.7%2B-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20Mac-lightgrey?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square"/>
</p>

# 🦋 Thyroid Disease Detection

> Predict thyroid conditions using Machine Learning and an interactive web app.

---

## 📑 Table of Contents
- [Demo](#demo)
- [Features](#features)
- [Technical Aspects](#technical-aspects)
- [Workflow](#workflow)
- [Data Pre-processing](#data-pre-processing)
- [Model Creation and Evaluation](#model-creation-and-evaluation)
- [Usage](#usage)
- [Author](#author)
- [Contributing](#contributing)

---

## 🚀 Demo & Deployment
[![Demo Video](https://img.shields.io/badge/Watch%20Demo-YouTube-red?logo=youtube&style=for-the-badge)](https://github.com/pavitra147/Thyroid-Disease-Detection/assets/130755029/1d78ea71-96e3-4e11-a746-2a8606b23c90)

[![Deploy App](https://img.shields.io/badge/Live%20App-Visit-green?style=for-the-badge&logo=heroku)](https://your-deployment-link.com)

---

## ✨ Features
- Predicts compensated, primary, secondary hypothyroid, or negative (no thyroid)
- Trained on UCI Machine Learning Repository dataset
- Uses XGBoost, Random Forest, and KNN models
- Web app built with Flask, HTML, and CSS
- Deployed on Heroku

---

## ⚙️ Technical Aspects
- **Python:** 3.7+
- **Libraries:** `sklearn`, `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Front-end:** HTML, CSS
- **Back-end:** Flask
- **IDE:** Jupyter Notebook, VSCode

---

## 🛠️ Workflow
- Dataset: [UCI Thyroid Disease Data Set](https://archive.ics.uci.edu/ml/datasets/thyroid+disease)

---

## 🧹 Data Pre-processing
- Missing values handled by KNN Imputer
- Outliers removed by boxplot/percentile methods
- Categorical features encoded (ordinal/label)
- Feature scaling with Standard Scaler
- Imbalanced data handled by SMOTE
- Dropped unnecessary columns

---

## 🤖 Model Creation and Evaluation
- Tested: Random Forest, XGBoost, KNN
- Final Model: XGBoost (after hyperparameter tuning with RandomizedSearchCV)
- Evaluated with accuracy, confusion matrix, classification report

---

## 💻 Usage
1. Clone the repo:
   ```powershell
   git clone https://github.com/manas1148/Thyroid-Disease-Detection.git
   cd Thyroid_disease
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Run the app:
   ```powershell
   python app.py
   ```
4. Open your browser at `http://localhost:5000`

---

## 👤 Author
- **Manas Rai Kaushik**  
  [LinkedIn](https://www.linkedin.com/in/manas-rai-kaushik-1b4200242/)

---

## 🤝 Contributing
If you find any bugs or have suggestions, please [raise an issue](#) or submit a pull request!

---

<p align="center">
  <b>⭐️ If you like this project, give it a star!</b>
</p>
