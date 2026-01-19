# 🩺 Pneumonia X-ray Image Classifier

A deep learning system for **automatic pneumonia detection from chest X-ray images**, built using transfer learning with MobileNetV2 and deployed as a FastAPI service inside a Docker container.

> ⚠️ **Disclaimer:** This project is for educational and demonstration purposes only.  
> It is **not a medical device** and must not be used for clinical decision-making.

---

## 📌 Project Overview

This project explores the end-to-end lifecycle of a machine learning system:

- Dataset analysis and preprocessing
- Model training and experimentation
- Evaluation on a held-out test set
- Deployment as an inference API
- Containerisation for reproducible deployment

The final model is optimised for **high recall**, making it suitable as a **screening tool** where missing pneumonia cases is more costly than false positives.

---

## 📊 Dataset

- **Source:** Chest X-Ray Images (Pneumonia) – Kaggle  
- **Classes:**  
  - `NORMAL`  
  - `PNEUMONIA`
- Images were resized and normalised using ImageNet statistics to match the pretrained backbone.

---

## 🧠 Model Architecture

- **Backbone:** MobileNetV2 (ImageNet pretrained)
- **Classifier Head:**
  - Global Average Pooling
  - Dropout (p = 0.5)
  - Linear layer (2 classes)
- **Training Strategy:**
  - Baseline: frozen backbone
  - Regularisation: dropout added
  - Final model: last 3 backbone blocks unfrozen (fine-tuning)
- **Loss:** Cross-Entropy
- **Optimizer:** Adam
- **Learning Rate (final):** 1e-4

---

## 🧪 Model Performance (Test Set)

Final evaluation performed **once** on a held-out test set using a standalone script.

| Metric (PNEUMONIA class) | Value |
|--------------------------|-------|
| Recall                   | **0.997** |
| Precision                | 0.720 |
| F1 Score                 | 0.837 |
| Accuracy                 | 0.756 |
| ROC-AUC                  | 0.939 |

**Interpretation:**
- The model detects nearly all pneumonia cases (very high recall).
- It produces false positives, which is expected given the recall-optimised objective.

---

## 🚀 API Usage

### Health Check
```bash
GET /health
````

### Prediction

```bash
POST /predict
```

**Input:**

* Multipart form upload with key `file` (JPEG/PNG image)

**Example:**

```bash
curl -X POST "http://localhost:7860/predict" \
  -H "accept: application/json" \
  -F "file=@xray.jpeg"
```

**Response:**

```json
{
  "prediction": "PNEUMONIA",
  "p_pneumonia": 0.9998,
  "note": "High-recall screening model; not a medical device."
}
```

---

## 🐳 Docker

### Build

```bash
docker build -t pneumonia-api .
```

### Run

```bash
docker run -p 7860:7860 pneumonia-api
```

The service will be available at:

```
http://localhost:7860
```

---

## 🧩 Key Learnings

* Transfer learning enables strong performance with limited data
* Validation metrics must guide model selection — not architecture choices alone
* High recall models trade specificity for safety
* Separating training, evaluation, and deployment improves reproducibility
* Containerisation simplifies real-world deployment

---

## 👤 Author

**Talha Akhoon**
Full-Stack / AI Engineer

---

## 📜 License

MIT License

