# 🤟 Sign Language Recognition System
**OpenCV + KNN · ASL A–Z · Streamlit**

---

## Project Structure
```
sign_language_recognition/
├── app.py                    # Streamlit web app
├── train_model.py            # Model training script
├── generate_sample_signs.py  # Synthetic dataset + sample sign generator
├── requirements.txt
├── utils/
│   └── preprocessing.py      # OpenCV preprocessing pipeline
├── dataset/
│   ├── train/  (A–Z folders, ~60 images each)
│   └── test/   (A–Z folders, ~15 images each)
├── model/
│   ├── knn_model.pkl
│   └── label_encoder.pkl
└── sample_signs/
    └── sign_A.png … sign_Z.png   ← upload these to test the app
```

---

## Quick Start

### 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### 2 — Generate dataset + sample signs
```bash
python generate_sample_signs.py
```
Creates ~1,950 augmented training images (26 letters × 60 variants + 15 test each).

### 3 — Train the model
```bash
python train_model.py
```
Prints accuracy + per-class report. Model saved to `model/`.

### 4 — Launch the app
```bash
streamlit run app.py
```

---

## Testing the Model
After running the app, go to the **Sample Signs** tab — download any `sign_X.png` image, then upload it in the **Predict** tab. The model will return:
- Predicted letter (A–Z)
- Confidence score (% of 5 KNN neighbours matching)
- Top-3 nearest-letter breakdown

---

## OpenCV Preprocessing Pipeline
| Step | Operation | Purpose |
|------|-----------|---------|
| 1 | Resize 64×64 | Uniform input size |
| 2 | BGR → Grayscale | Reduce channels |
| 3 | CLAHE | Contrast normalisation |
| 4 | Gaussian Blur (3×3) | Noise removal |
| 5 | Normalize → Flatten | Feature vector for KNN |

---

## Tech Stack
- **OpenCV** — image preprocessing
- **scikit-learn KNN** — classification
- **NumPy** — array ops & augmentation
- **Streamlit** — web UI
- **Python 3.10+**

---

## Portfolio Notes
- End-to-end pipeline: data generation → preprocessing → training → deployment
- 26-class multi-class classifier with ~95%+ accuracy on synthetic data
- Visualises every preprocessing stage in the UI
- Easily swappable with real ASL dataset (Kaggle ASL Alphabet) by replacing `dataset/`
