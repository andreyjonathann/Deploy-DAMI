import streamlit as st
import numpy as np
import pandas as pd
import joblib
import re
import os
import gdown
from PIL import Image
import tf_keras as tf
from tf_keras import layers, models

# ─── Konfigurasi Halaman ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pneumonia Clustering",
    page_icon="🫁",
    layout="centered"
)

# ─── CSS Kustom ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .main { background-color: #0f1117; }
    .block-container { padding-top: 2rem; max-width: 800px; }
    h1, h2, h3 { font-family: 'Space Mono', monospace; }

    .header-title {
        font-family: 'Space Mono', monospace;
        font-size: 2rem; font-weight: 700;
        color: #00d4ff; letter-spacing: -1px; margin-bottom: 0;
    }
    .header-sub {
        color: #8892a4; font-size: 0.95rem;
        margin-top: 0.2rem; margin-bottom: 2rem;
    }
    .result-card {
        background: linear-gradient(135deg, #1a1f2e, #141824);
        border: 1px solid #2a3044; border-left: 4px solid #00d4ff;
        border-radius: 12px; padding: 1.5rem 2rem; margin-top: 1.5rem;
    }
    .result-card h3 {
        color: #00d4ff; font-size: 1rem; margin-bottom: 1rem;
        text-transform: uppercase; letter-spacing: 2px;
    }
    .cluster-badge {
        display: inline-block; background: #00d4ff22; color: #00d4ff;
        border: 1px solid #00d4ff55; border-radius: 8px;
        padding: 0.3rem 0.8rem; font-family: 'Space Mono', monospace;
        font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;
    }
    .interp-text { color: #e2e8f0; font-size: 1.1rem; font-weight: 500; margin: 0.5rem 0 1rem 0; }
    .feature-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.8rem; }
    .feature-pill {
        background: #1e2536; border: 1px solid #2d3650;
        border-radius: 20px; padding: 0.25rem 0.75rem;
        font-size: 0.8rem; color: #94a3b8;
    }
    .feature-pill.active { background: #00d4ff18; border-color: #00d4ff55; color: #00d4ff; }
    .info-row {
        display: flex; justify-content: space-between;
        border-top: 1px solid #2a3044; padding-top: 1rem; margin-top: 1rem;
    }
    .info-item label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; display: block; }
    .info-item span { color: #e2e8f0; font-family: 'Space Mono', monospace; font-size: 0.9rem; }
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #0099cc);
        color: #0f1117; border: none; border-radius: 8px;
        font-family: 'Space Mono', monospace; font-weight: 700;
        font-size: 0.9rem; padding: 0.6rem 2rem; width: 100%;
    }
    .stButton > button:hover { opacity: 0.85; color: #0f1117; }
    .stTextArea textarea { background: #1a1f2e; border: 1px solid #2a3044; border-radius: 8px; color: #e2e8f0; }
    .divider { border: none; border-top: 1px solid #2a3044; margin: 1.5rem 0; }
    .error-box { background: #2d1b1b; border: 1px solid #ff4444; border-radius: 8px; padding: 1rem; color: #ff8080; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ─── Download Model dari Google Drive ─────────────────────────────────────────
MODEL_DIR     = "model"
GDRIVE_FOLDER_ID = "1QJVa0RN6E_lDaBqWrOaBfzFGJoE2mwf6"

@st.cache_resource
def download_and_load_models():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Cek apakah model sudah ada (supaya tidak download ulang)
    if not os.path.exists(os.path.join(MODEL_DIR, "config_deploy.pkl")):
        with st.spinner("⏬ Mendownload model dari Google Drive (hanya sekali)..."):
            try:
                url = f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}"
                gdown.download_folder(url, output=MODEL_DIR, quiet=False, use_cookies=False)
            except Exception as e:
                return None, f"Gagal download model: {e}"

    # Cari subfolder jika gdown membuat subfolder di dalam MODEL_DIR
    actual_dir = MODEL_DIR
    for item in os.listdir(MODEL_DIR):
        item_path = os.path.join(MODEL_DIR, item)
        if os.path.isdir(item_path):
            if os.path.exists(os.path.join(item_path, "config_deploy.pkl")):
                actual_dir = item_path
                break

    try:
        encoder         = tf.keras.models.load_model(os.path.join(actual_dir, "encoder_xray.keras"))
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
        return None, f"Gagal load model: {e}"

# ─── Helper Functions ─────────────────────────────────────────────────────────
NEGATION_PATTERNS = [
    r"tidak tampak", r"tidak terlihat", r"tidak ada",
    r"tanpa", r"tak tampak", r"no evidence of", r"negative for"
]

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
        "fitur_bronchopneumonia": has_positive_keyword(t, [r"bronchopneumonia", r"bronkopneumonia", r"broncho pneumonia", r"bronko pneumonia"]),
        "fitur_fibrotic":         has_positive_keyword(t, [r"fibrotic", r"fibrotik", r"fibrosis"]),
        "fitur_efusi":            has_positive_keyword(t, [r"efusi", r"pleural effusion", r"effusion"]),
        "fitur_bilateral":        has_positive_keyword(t, [r"bilateral", r"kedua paru"]),
        "fitur_pelebaran":        has_positive_keyword(t, [r"pelebaran", r"dilatasi", r"kardiomegali", r"cardiomegaly", r"cor membesar"]),
        "fitur_normal":           has_positive_keyword(t, [r"normal", r"dalam batas normal", r"tidak tampak kelainan"]),
    })

def get_report_category(row):
    if row["fitur_normal"] == 1 and row["fitur_pneumonia"] == 0 and row["fitur_bronchopneumonia"] == 0:
        return "Normal"
    if row["fitur_bronchopneumonia"] == 1: return "Bronchopneumonia"
    if row["fitur_fibrotic"] == 1:         return "Fibrotic Pneumonia"
    if row["fitur_efusi"] == 1 and row["fitur_bilateral"] == 1: return "Pneumonia Bilateral Efusi"
    if row["fitur_bilateral"] == 1:        return "Pneumonia Bilateral"
    if row["fitur_efusi"] == 1:            return "Pneumonia Efusi"
    if row["fitur_pelebaran"] == 1 and row["fitur_pneumonia"] == 1: return "Pneumonia dengan Pelebaran"
    if row["fitur_pneumonia"] == 1:        return "Pneumonia"
    return "Lainnya"

CLINICAL_COLS = ["fitur_pneumonia", "fitur_bronchopneumonia", "fitur_fibrotic",
                 "fitur_efusi", "fitur_bilateral", "fitur_pelebaran", "fitur_normal"]

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<p class="header-title">🫁 Pneumonia Clustering</p>', unsafe_allow_html=True)
st.markdown('<p class="header-sub">Sistem Klasifikasi Multimodal — Citra X-Ray + Laporan Radiologi</p>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ─── Load Model ───────────────────────────────────────────────────────────────
result, err = download_and_load_models()
if result is None:
    st.markdown(f'<div class="error-box">❌ {err}</div>', unsafe_allow_html=True)
    st.stop()

encoder, latent_scaler, pca_img, clinical_scaler, onehot, tfidf, svd_text, tfidf_scaler, kmeans, config = result
best_config            = config["best_config"]
cluster_interpretation = config["cluster_interpretation"]
IMG_SIZE               = config["img_size"]
best_score             = config["best_score"]

st.success("✅ Model berhasil dimuat!")

# ─── Input ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.markdown("**📁 Upload Gambar X-Ray**")
    uploaded_file = st.file_uploader("Format: JPG, PNG", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded_file:
        st.image(Image.open(uploaded_file), caption="Preview X-Ray", use_container_width=True)

with col2:
    st.markdown("**📝 Laporan Radiologi**")
    report_text = st.text_area(
        "Laporan", height=200,
        placeholder="Contoh:\nTampak infiltrat pada lapang paru kanan.\n\nKesan: Pneumonia lobaris kanan.",
        label_visibility="collapsed"
    )

st.markdown("")
predict_btn = st.button("🔍 Analisis Cluster")

# ─── Prediksi ─────────────────────────────────────────────────────────────────
if predict_btn:
    if uploaded_file is None:
        st.warning("⚠️ Silakan upload gambar X-Ray terlebih dahulu.")
    elif not report_text.strip():
        st.warning("⚠️ Silakan masukkan laporan radiologi.")
    else:
        with st.spinner("Memproses..."):
            try:
                img   = Image.open(uploaded_file).convert("L").resize((IMG_SIZE, IMG_SIZE))
                arr   = np.expand_dims(np.expand_dims(np.array(img).astype("float32") / 255.0, -1), 0)

                latent_new_pca      = pca_img.transform(latent_scaler.transform(encoder.predict(arr, verbose=0)))
                feat_series         = extract_report_features(report_text)
                clinical_new_scaled = clinical_scaler.transform(feat_series[CLINICAL_COLS].astype(float).values.reshape(1, -1))

                temp_row = feat_series.copy()
                temp_row["teks_laporan"] = clean_text(report_text)
                category_new        = get_report_category(temp_row)
                category_new_onehot = onehot.transform(pd.DataFrame({"kategori_report": [category_new]}))

                tfidf_new_scaled = tfidf_scaler.transform(svd_text.transform(tfidf.transform([clean_text(report_text)])))

                iw, cw, catw, tw = best_config["image_weight"], best_config["clinical_weight"], best_config["category_weight"], best_config["tfidf_weight"]

                final_vector = np.hstack([latent_new_pca * iw, clinical_new_scaled * cw, category_new_onehot * catw, tfidf_new_scaled * tw])
                cluster_pred = int(kmeans.predict(final_vector)[0])
                interpretasi = cluster_interpretation.get(cluster_pred, "Interpretasi belum tersedia")

                active   = [c.replace("fitur_", "").replace("_", " ").title() for c in CLINICAL_COLS if feat_series[c] == 1]
                inactive = [c.replace("fitur_", "").replace("_", " ").title() for c in CLINICAL_COLS if feat_series[c] == 0]
                pills    = "".join([f'<span class="feature-pill active">✓ {f}</span>' for f in active])
                pills   += "".join([f'<span class="feature-pill">✗ {f}</span>' for f in inactive])

                st.markdown(f"""
                <div class="result-card">
                    <h3>Hasil Analisis</h3>
                    <div class="cluster-badge">Cluster {cluster_pred}</div>
                    <p class="interp-text">{interpretasi}</p>
                    <div style="color:#64748b;font-size:0.85rem;margin-bottom:0.5rem;">
                        Kategori: <span style="color:#e2e8f0">{category_new}</span>
                    </div>
                    <div style="color:#64748b;font-size:0.8rem;margin-bottom:0.3rem;">Fitur Klinis:</div>
                    <div class="feature-row">{pills}</div>
                    <div class="info-row">
                        <div class="info-item"><label>Silhouette</label><span>{best_score:.4f}</span></div>
                        <div class="info-item"><label>Image Weight</label><span>{iw}</span></div>
                        <div class="info-item"><label>K Optimal</label><span>{best_config['k']}</span></div>
                        <div class="info-item"><label>Dimensi</label><span>{best_config['dimensi_vector']}</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.markdown(f'<div class="error-box">❌ Error: {e}</div>', unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<p style="color:#2a3044;font-size:0.75rem;text-align:center;font-family:Space Mono,monospace;">Pneumonia Multimodal Clustering System</p>', unsafe_allow_html=True)
