# VeritasAI: Fake News Detection System 🧠📰

VeritasAI is a machine learning-powered web app designed to detect fake news using natural language processing (NLP). Built with **Streamlit**, it allows users to input news headlines or articles and get real-time predictions on whether the content is fake or real.

---

## 🚀 Features

- Detects fake vs. real news using SVM and PassiveAggressiveClassifier models
- Cleaned and preprocessed textual data with TF-IDF vectorization
- Streamlit-based interactive web interface
- Confidence score for each prediction
- Exportable chat interface (optional for future upgrades)

---

## 📂 Files in This Repo

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit app |
| `veritasAI_model.pkl` | Trained ML model (SVM) |
| `veritasAI_vectorizer.pkl` | Trained TF-IDF vectorizer |
| `requirements.txt` | Python dependencies |

---

## 🧠 Model Details

- Dataset: COVID19-FNIR (Fake and Real News Dataset)
- Vectorization: TF-IDF
- ML Models: 
  - PassiveAggressiveClassifier: 92.03% accuracy  
  - Support Vector Machine (SVM): **94.14% accuracy** ✅

---

## 💻 How to Run

1. Install requirements:

   ```bash
   pip install -r requirements.txt
