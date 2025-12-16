# Twitter Sentiment Analysis 🐦📊

A complete **end-to-end NLP project** that classifies tweets into  
**Positive, Negative, Neutral, or Irrelevant** sentiments using **Deep Learning (LSTM)** and serves predictions through a **Flask web app**.

---

## 🚀 Features

- Multi-class sentiment classification (4 classes)
- Deep Learning model built with **TensorFlow / Keras**
- **Data-driven tokenization** (`num_words`, `maxlen`)
- Clean text preprocessing pipeline
- Flask-based web interface for real-time predictions
- Styled UI with color-coded sentiment output
- Production-ready model loading (`.keras` format)

---

## 🧠 Model Overview

- **Architecture:**  
  `Embedding → BiLSTM → Dense (Softmax)`
- **Loss:** `sparse_categorical_crossentropy`
- **Optimizer:** Adam
- **Classes:**  
  - Positive  
  - Negative  
  - Neutral  
  - Irrelevant  

---

## 🗂 Project Structure

Twitter Sentiment Analysis/
│
|── experiments/Data/EDA and model_training.ipynb
├── app.py # Flask application
├── model.keras # Trained sentiment model
├── tokenizer.pkl # Fitted tokenizer
├── requirements.txt # Dependencies
│
├── templates/
│ └── home.html # Web UI
│
└── README.md

## To Run
pip install -r requirements.txt
python app.py
