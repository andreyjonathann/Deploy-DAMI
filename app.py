import streamlit as st
import numpy as np
import pandas as pd
import joblib
import re
import os
import gdown
from PIL import Image
import tensorflow as tf


# ─────────────────────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pneumonia Clustering",
    page_icon="🫁",
    layout="centered"
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.main { background-color: #0f1117; }
.block-container { padding-top: 2rem; max-width: 800px; }

.header-title {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #00d4ff;
    margin-bottom: 0;
}
.header-sub {
    color: #8892a4;
    font-size: 0.95rem;
    margin-top: 0.2rem;
    margin-bottom: 2rem;
}
.result-card {
    background: linear-gradient(135deg, #1a1f2e, #141824);
    border: 1px solid #2a3044;
    border-left: 4px solid #00d4ff;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-top: 1.5rem;
}
.result-card h3 {
    color: #00d4ff;
    font-size: 1rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 1rem;
}
.cluster-badge {
    display: inline-block;
    background: #00d4ff22;
    color: #00d4ff;
    border: 1px solid #00d4ff55;
    border-radius: 8px;
    padding: 0.3rem 0.9rem;
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 0.8rem;
}
.interp-text {
    color: #e2e8f0;
    font-size: 1.05rem;
    font-weight: 500;
    margin: 0.4rem 0 1rem 0;
}
.section-label {
    color: #64748b;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.4rem;
}
.category-value {
    color: #e2e8f0;
    font-size: 1rem;
    margin-bottom: 1rem;
}
.feature-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.4rem;
    margin-top: 0.5rem;
    margin-bottom: 1rem;
}
.feature-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: #1e2536;
    border: 1px solid #2d3650;
    border-radius: 8px;
    padding: 0.4rem 0.7rem;
    font-size: 0.85rem;
    color: #94a3b8;
}
.feature-item.active {
    background: #00d4ff12;
    border-color: #00d4ff44;
    color: #e2e8f0;
}
.feature-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #2d3650;
    flex-shrink: 0;
}
.feature-dot.active {
    background: #00d4ff;
}
.info-row {
    display: flex;
    justify-content: space-between;
    border-top: 1px solid #2a3044;
    padding-top: 1rem;
    margin-top: 0.5rem;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.info-item label {
    font-size: 0.7rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    display: block;
    margin-bottom: 0.2rem;
}
.info-item span {
    color: #e2e8f0;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
}
.error-box {
    background: #2d1b1b;
    border: 1px solid #ff4444;
    border-radius: 8px;
    padding: 1rem;
    color: #ff8080;
    font-size: 0.9rem;
}
.divider {
    border: none;
    border-top: 1px solid #2a3044;
    margin: 1.5rem 0;
}
.stButton > button {
    background: linear-gradient(135deg, #00d4ff, #0099cc);
    color: #0f1117;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 0.9rem;
    padding: 0.6rem 2rem;
    width: 100%;
}
.stButton > button:hover { opacity: 0.85; color: #0f1117; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONFIG MODEL
# ─────────────────────────────────────────────────────────────
MODEL_DIR = "model"
GDRIVE_FOLDER_ID = "1QJVa0RN6E_lDaBqWrOaBfzFGJoE2mwf6"


def build_encoder():
    model = tf.Sequential([
        tf.layers.Input(shape=(128, 128, 1)),
        tf.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.layers.BatchNormalization(),
        tf.layers.MaxPooling2D((2, 2), padding="same"),
        tf.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.layers.BatchNormalization(),
        tf.layers.MaxPooling2D((2, 2), padding="same"),
        tf.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        tf.layers.BatchNormalization(),
        tf.layers.MaxPooling2D((2, 2), padding="same"),
        tf.layers.Flatten(),
        tf.layers.Dense(128, activation="linear", name="latent_vector")
    ])
    return model


# ─────────────────────────────────────────────────────────────
# DOWNLOAD + LOAD MODEL
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def download_and_load_models():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(os.path.join(MODEL_DIR, "config_deploy.pkl")):
        with st.spinner("⏬ Mendownload model dari Google Drive (hanya sekali)..."):
            try:
                url = f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}"
                gdown.download_folder(url, output=MODEL_DIR, quiet=False, use_cookies=False)
            except Exception as e:
                return None, f"Gagal download model: {e}"

    actual_dir = MODEL_DIR
    for item in os.listdir(MODEL_DIR):
        item_path = os.path.join(MODEL_DIR, item)
        if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "config_deploy.pkl")):
            actual_dir = item_path
            break

    try:
        encoder = build_encoder()
        encoder.load_weights(os.path.join(actual_dir, "model_encoder.h5"))

        latent_scaler   = joblib.load(os.path.join(actual_dir, "latent_scaler.pkl"))
        pca_img         = joblib.load(os.path.join(actual_dir, "pca_img.pkl"))
        clinical_scaler = joblib.load(os.path.join(actual_dir, "clinical_scaler.pkl"))
        onehot          = joblib.load(os.path.join(actual_dir, "onehot_category.pkl"))
        tfidf           = joblib.load(os.path.join(actual_dir, "tfidf_vectorizer.pkl"))
        svd_text        = joblib.load(os.path.join(actual_dir, "svd_text.pkl"))
        tfidf_scaler    = joblib.load(os.path.join(actual_dir, "tfidf_scaler.pkl"))
        kmeans          = joblib.load(os.path.join(actual_dir, "kmeans_model.pkl"))
        config          = joblib.load(os.path.join(actual_dir, "config_deploy.pkl"))

        return (encoder, latent_scaler, pca_img, clinical_scaler,
                onehot, tfidf, svd_text, tfidf_scaler, kmeans, config), None
    except Exception as e:
        return None, f"Gagal load model: {str(e)}"


# ─────────────────────────────────────────────────────────────
# TEXT PROCESSING
# ─────────────────────────────────────────────────────────────
NEGATION_PATTERNS = [
    r"tidak tampak", r"tidak terlihat", r"tidak ada",
    r"tanpa", r"tak tampak", r"no evidence of", r"negative for"
]

CLINICAL_COLS = [
    "fitur_pneumonia", "fitur_bronchopneumonia", "fitur_fibrotic",
    "fitur_efusi", "fitur_bilateral", "fitur_pelebaran", "fitur_normal"
]

CLINICAL_LABELS = {
    "fitur_pneumonia":        "Pneumonia",
    "fitur_bronchopneumonia": "Bronchopneumonia",
    "fitur_fibrotic":         "Fibrotic",
    "fitur_efusi":            "Efusi",
    "fitur_bilateral":        "Bilateral",
    "fitur_pelebaran":        "Pelebaran",
    "fitur_normal":           "Normal",
}


def clean_text(text):
    if pd.isna(text): return ""
    text = str(text).lower().replace("\n", " ")
    text = re.sub(r"[^a-zA-Z0-9\s\-\/]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_negated(text, keyword):
    for neg in NEGATION_PATTERNS:
        if re.search(rf"{neg}\s+[\w\s]{{0,40}}{keyword}", clean_text(text)):
            return True
    return False


def has_positive_keyword(text, keywords):
    text = clean_text(text)
    for kw in keywords:
        if re.search(kw, text) and not is_negated(text, kw):
            return 1
    return 0


def extract_report_features(text):
    t = clean_text(text)
    return pd.Series({
        "fitur_pneumonia":        has_positive_keyword(t, [r"pneumonia", r"infiltrat", r"konsolidasi", r"consolidation"]),
        "fitur_bronchopneumonia": has_positive_keyword(t, [r"bronchopneumonia", r"bronkopneumonia"]),
        "fitur_fibrotic":         has_positive_keyword(t, [r"fibrosis", r"fibrotic"]),
        "fitur_efusi":            has_positive_keyword(t, [r"efusi", r"effusion"]),
        "fitur_bilateral":        has_positive_keyword(t, [r"bilateral"]),
        "fitur_pelebaran":        has_positive_keyword(t, [r"kardiomegali", r"cardiomegaly"]),
        "fitur_normal":           has_positive_keyword(t, [r"normal"]),
    })


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown('<p class="header-title">🫁 Pneumonia Clustering</p>', unsafe_allow_html=True)
st.markdown('<p class="header-sub">Sistem Klasifikasi Multimodal — Citra X-Ray + Laporan Radiologi</p>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────
result, err = download_and_load_models()

if result is None:
    st.markdown(f'<div class="error-box">❌ {err}</div>', unsafe_allow_html=True)
    st.stop()

encoder, latent_scaler, pca_img, clinical_scaler, onehot, tfidf, svd_text, tfidf_scaler, kmeans, config = result
IMG_SIZE    = config["img_size"]
best_config = config["best_config"]

st.success("✅ Model berhasil dimuat!")

# ─────────────────────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.markdown("**📁 Upload Gambar X-Ray**")
    uploaded_file = st.file_uploader("Format: JPG, PNG", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded_file:
        st.image(Image.open(uploaded_file), caption="Preview X-Ray", use_container_width=True)

with col2:
    st.markdown("**📝 Laporan Radiologi**")
    report_text = st.text_area(
        "Laporan",
        height=200,
        placeholder="Contoh:\nTampak infiltrat pada lapang paru kanan.\n\nKesan: Pneumonia lobaris kanan.",
        label_visibility="collapsed"
    )

st.markdown("")
predict_btn = st.button("🔍 Analisis Cluster")

# ─────────────────────────────────────────────────────────────
# PREDIKSI
# ─────────────────────────────────────────────────────────────
if predict_btn:
    if uploaded_file is None:
        st.warning("⚠️ Silakan upload gambar X-Ray terlebih dahulu.")
        st.stop()
    if not report_text.strip():
        st.warning("⚠️ Silakan masukkan laporan radiologi.")
        st.stop()

    with st.spinner("Memproses..."):
        try:
            # Preprocess gambar
            img = Image.open(uploaded_file).convert("L").resize((IMG_SIZE, IMG_SIZE))
            arr = np.expand_dims(np.expand_dims(np.array(img).astype("float32") / 255.0, -1), 0)

            # Latent vector
            latent     = encoder.predict(arr, verbose=0)
            latent_pca = pca_img.transform(latent_scaler.transform(latent))

            # Fitur klinis
            feat_series     = extract_report_features(report_text)
            clinical_scaled = clinical_scaler.transform(feat_series[CLINICAL_COLS].astype(float).values.reshape(1, -1))

            # Kategori
            category_text = "Normal"
            if feat_series["fitur_bronchopneumonia"] == 1: category_text = "Bronchopneumonia"
            elif feat_series["fitur_fibrotic"] == 1:       category_text = "Fibrotic Pneumonia"
            elif feat_series["fitur_pneumonia"] == 1:      category_text = "Pneumonia"
            category_onehot = onehot.transform(pd.DataFrame({"kategori_report": [category_text]}))

            # TF-IDF
            tfidf_scaled = tfidf_scaler.transform(svd_text.transform(tfidf.transform([clean_text(report_text)])))

            # Gabung & prediksi
            iw, cw, catw, tw = best_config["image_weight"], best_config["clinical_weight"], best_config["category_weight"], best_config["tfidf_weight"]
            final_vector = np.hstack([latent_pca * iw, clinical_scaled * cw, category_onehot * catw, tfidf_scaled * tw])
            cluster_pred = int(kmeans.predict(final_vector)[0])
            interpretasi = config["cluster_interpretation"].get(cluster_pred, "Interpretasi belum tersedia")

            # ── Build feature grid HTML ──────────────────────────────
            feature_items = ""
            for col in CLINICAL_COLS:
                is_active  = feat_series[col] == 1
                cls        = "active" if is_active else ""
                dot_cls    = "active" if is_active else ""
                label      = CLINICAL_LABELS[col]
                icon       = "✓" if is_active else "✗"
                feature_items += f"""
                <div class="feature-item {cls}">
                    <div class="feature-dot {dot_cls}"></div>
                    {icon} {label}
                </div>"""

            # ── Render hasil ─────────────────────────────────────────
            st.markdown(f"""
            <div class="result-card">
                <h3>Hasil Analisis</h3>

                <div class="cluster-badge">Cluster {cluster_pred}</div>

                <p class="interp-text">{interpretasi}</p>

                <div class="section-label">Kategori Report Terdeteksi</div>
                <div class="category-value">{category_text}</div>

                <div class="section-label">Fitur Klinis</div>
                <div class="feature-grid">{feature_items}</div>

                <div class="info-row">
                    <div class="info-item">
                        <label>Silhouette Score</label>
                        <span>{config["best_score"]:.4f}</span>
                    </div>
                    <div class="info-item">
                        <label>Image Weight</label>
                        <span>{iw}</span>
                    </div>
                    <div class="info-item">
                        <label>K Optimal</label>
                        <span>{best_config['k']}</span>
                    </div>
                    <div class="info-item">
                        <label>Dimensi Vektor</label>
                        <span>{best_config['dimensi_vector']}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.markdown(f'<div class="error-box">❌ Error saat prediksi: {e}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown(
    '<p style="color:#2a3044;font-size:0.75rem;text-align:center;font-family:Space Mono,monospace;">Pneumonia Multimodal Clustering System</p>',
    unsafe_allow_html=True
)
