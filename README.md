# Encoder-Only Transformer from Scratch for Sentiment Analysis

## 🧠 Overview

This project demonstrates the implementation of a **custom encoder-only Transformer model from scratch** using TensorFlow and applies it to perform **sentiment analysis** on the **Amazon Fine Food Reviews** dataset.

This implementation was created to deeply understand:
- The mechanics of self-attention
- Positional encoding
- Feed-forward networks
- Building a complete transformer block without using any prebuilt libraries like HuggingFace Transformers

---

## 📁 Project Structure
├── Custom Transformer.ipynb # Jupyter notebook with end-to-end implementation

├── README.md # This file

---

## 📌 Problem Statement

To develop a transformer-based model from scratch (encoder-only) to predict sentiment (positive/negative) from textual food reviews on Amazon.

---

## 📦 Dataset: Amazon Fine Food Reviews

The dataset contains over 500,000 reviews of fine food products on Amazon.

- Total Reviews: 568,454
- Users: 256,059
- Products: 74,258
- Ratings: 1 to 5 (mapped to binary for sentiment analysis)

Dataset publicly available on Kaggle.

---

## ⚙️ Technical Stack

- Python 3.12.3
- TensorFlow 2.x
- NumPy, pandas, scikit-learn
- Matplotlib, Seaborn
- Custom layers, attention, and transformer encoder blocks

---

## 🛠️ Model Architecture

### ✅ Encoder-Only Transformer

- Positional Encoding
- Scaled Dot-Product Attention
- Multi-Head Attention
- Layer Normalization
- Feed Forward Network
- Residual Connections

### 🧱 Model Pipeline

1. Data Preprocessing:
   - Cleaning, tokenization, padding
   - Label conversion for binary sentiment (e.g., 1 for positive, 0 for negative)

2. Custom Positional Encoding

3. Multi-Head Self-Attention

4. Feed Forward Network

5. Final Dense Output Layer

---

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

| Metric    | Score     |
|-----------|-----------|
| Accuracy  | ~83–87%   |
| F1-Score  | ~0.84     |

---

## 📈 Training Outputs

- Accuracy and loss graphs across epochs
- Sample review predictions
- Confusion matrix visualizations

---

## 🧪 Experiments You Can Try

- Increase or decrease encoder layers (depth)
- Change number of attention heads
- Modify hidden dimension sizes
- Try both sinusoidal and learnable positional encodings

---

## ✅ Ethical Note

- Data used is publicly available and does not include any personally identifiable information.
- This project is intended solely for educational and research purposes.

---

## 📚 References

- Vaswani et al. (2017), "Attention is All You Need"
- TensorFlow official documentation
- Course and academic literature on NLP and transformers

---

## 🙌 Acknowledgment

Built as part of a hands-on learning exercise to understand the internals of Transformer models and apply them to real-world NLP tasks.

---
