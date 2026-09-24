"""
app.py  –  Sign Language Recognition · Streamlit UI
Run:  streamlit run app.py
"""

import os, sys, pickle
import numpy as np
import cv2
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from utils.preprocessing import preprocess, preprocess_visual

BASE   = os.path.dirname(__file__)
MODEL  = os.path.join(BASE, "model", "knn_model.pkl")
ENC    = os.path.join(BASE, "model", "label_encoder.pkl")
ACC_F  = os.path.join(BASE, "model", "accuracy.txt")
SIGNS  = os.path.join(BASE, "sample_signs")

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ASL Sign Recognizer",
    page_icon="🤟",
    layout="wide",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

.main { background: #0d0f14; }
[data-testid="stAppViewContainer"] { background: #0d0f14; color: #e8eaf2; }
[data-testid="stSidebar"] { background: #13161f; border-right: 1px solid #1e2130; }

h1 { 
  font-size: 2.4rem; font-weight: 700; letter-spacing: -0.5px;
  background: linear-gradient(135deg, #7c6fff 0%, #4fc3f7 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
h2, h3 { color: #c8cbdc; }

.result-box {
  background: linear-gradient(135deg, #1a1d2e 0%, #1e2235 100%);
  border: 1px solid #7c6fff44;
  border-radius: 16px;
  padding: 28px 32px;
  text-align: center;
  margin-top: 12px;
}
.pred-letter {
  font-size: 6rem; font-weight: 700; line-height: 1;
  background: linear-gradient(135deg, #7c6fff, #4fc3f7);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.pred-conf {
  font-size: 1.1rem; color: #8b93b5; margin-top: 6px;
}
.sign-tile img { border-radius: 8px; border: 2px solid #1e2130; }
.sign-tile:hover img { border-color: #7c6fff; }

.step-badge {
  display: inline-block;
  background: #7c6fff22;
  color: #a89dff;
  border-radius: 6px;
  padding: 2px 10px;
  font-size: 0.75rem;
  font-weight: 600;
  margin-bottom: 6px;
  letter-spacing: 0.5px;
}
.acc-chip {
  display: inline-block;
  background: #4fc3f722;
  color: #4fc3f7;
  border-radius: 20px;
  padding: 4px 14px;
  font-size: 0.85rem;
  font-weight: 600;
}
stButton > button {
  background: linear-gradient(135deg,#7c6fff,#4fc3f7) !important;
  color:#fff !important; border:none !important; border-radius:8px !important;
}
</style>
""", unsafe_allow_html=True)

# ── load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL):
        return None, None
    with open(MODEL, "rb") as f:
        clf = pickle.load(f)
    with open(ENC, "rb") as f:
        le = pickle.load(f)
    return clf, le

clf, le = load_model()

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤟 ASL Recognizer")
    st.markdown("---")

    trained_acc = ""
    if os.path.exists(ACC_F):
        trained_acc = open(ACC_F).read().strip()

    if clf:
        st.success("Model loaded ✓")
        if trained_acc:
            st.markdown(f"**Test accuracy:** <span class='acc-chip'>{trained_acc}%</span>",
                        unsafe_allow_html=True)
    else:
        st.error("No model found. Run `train_model.py` first.")

    st.markdown("---")
    st.markdown("### Pipeline")
    for step in ["1 · Resize to 64×64",
                 "2 · Grayscale",
                 "3 · CLAHE equalise",
                 "4 · Gaussian blur",
                 "5 · Flatten → KNN"]:
        st.markdown(f"<span class='step-badge'>{step}</span>", unsafe_allow_html=True)
        st.markdown("")

    st.markdown("---")
    st.markdown("### About")
    st.caption("26-class ASL letter recognition using OpenCV preprocessing + KNN.")

# ── main layout ───────────────────────────────────────────────────────────────
st.markdown("# 🤟 Sign Language Recognition")
st.markdown("Upload any ASL hand-sign image and the model will predict the letter.")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📚 Sample Signs", "📊 Preprocessing"])

# ── TAB 1 : Predict ───────────────────────────────────────────────────────────
with tab1:
    col_up, col_res = st.columns([1, 1], gap="large")

    with col_up:
        st.markdown("### Upload Image")
        uploaded = st.file_uploader("Choose a PNG / JPG sign image",
                                    type=["png", "jpg", "jpeg"])
        if uploaded:
            pil = Image.open(uploaded).convert("RGB")
            st.image(pil, caption="Uploaded image", use_container_width=True)

    with col_res:
        st.markdown("### Prediction")
        if uploaded and clf:
            img_np = np.array(pil)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            feat = preprocess(img_bgr)
            probs_idx = clf.kneighbors([feat], n_neighbors=5, return_distance=True)
            dists, idxs = probs_idx
            y_pred = clf.predict([feat])[0]
            letter = le.inverse_transform([y_pred])[0]

            # confidence = fraction of k-neighbours matching prediction
            neighbour_labels = clf._y[idxs[0]]
            conf = int(np.mean(neighbour_labels == y_pred) * 100)

            st.markdown(f"""
            <div class="result-box">
              <div class="pred-letter">{letter}</div>
              <div class="pred-conf">Confidence: {conf}% &nbsp;·&nbsp; KNN (k=5)</div>
            </div>
            """, unsafe_allow_html=True)

            # top-3 guesses
            from collections import Counter
            counts = Counter(le.inverse_transform(neighbour_labels))
            st.markdown("#### Top neighbours")
            for lbl, cnt in counts.most_common(3):
                st.progress(cnt / 5, text=f"{lbl}  ({cnt}/5 neighbours)")

        elif uploaded and not clf:
            st.warning("Train the model first: `python train_model.py`")
        else:
            st.info("Upload an image to get a prediction.")

# ── TAB 2 : Sample Signs ──────────────────────────────────────────────────────
with tab2:
    st.markdown("### ASL Sample Signs (A–Z)")
    st.caption("Download any image below and upload it in the **Predict** tab.")

    if os.path.isdir(SIGNS):
        letters = sorted([f for f in os.listdir(SIGNS) if f.endswith(".png")])
        cols = st.columns(9)
        for i, fname in enumerate(letters):
            letter = fname.replace("sign_", "").replace(".png", "")
            path = os.path.join(SIGNS, fname)
            with cols[i % 9]:
                st.image(path, caption=letter, use_container_width=True)
    else:
        st.warning("Run `python generate_sample_signs.py` to create sample signs.")

# ── TAB 3 : Preprocessing ─────────────────────────────────────────────────────
with tab3:
    st.markdown("### Visualise Preprocessing Pipeline")
    st.caption("Upload an image to see each stage of the OpenCV pipeline.")

    up2 = st.file_uploader("Choose image for pipeline view",
                            type=["png", "jpg", "jpeg"], key="pipe")
    if up2:
        pil2   = Image.open(up2).convert("RGB")
        img_np = np.array(pil2)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        stages = preprocess_visual(img_bgr)

        stage_cols = st.columns(len(stages))
        for col, (name, stage_img) in zip(stage_cols, stages.items()):
            with col:
                st.image(stage_img, caption=name, use_container_width=True)
    else:
        st.info("Upload an image above to visualise each preprocessing stage.")
