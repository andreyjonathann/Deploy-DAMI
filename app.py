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

.main {
    background-color: #0f1117;
}

.block-container {
    padding-top: 2rem;
    max-width: 800px;
}

.header-title {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #00d4ff;
}

.header-sub {
    color: #8892a4;
    font-size: 0.95rem;
    margin-bottom: 2rem;
}

.result-card {
    background: linear-gradient(135deg, #1a1f2e, #141824);
    border: 1px solid #2a3044;
    border-left: 4px solid #00d4ff;
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 1.5rem;
}

.cluster-badge {
    display: inline-block;
    background: #00d4ff22;
    color: #00d4ff;
    border-radius: 8px;
    padding: 0.4rem 0.8rem;
    font-size: 1.4rem;
    font-weight: bold;
}

.error-box {
    background: #2d1b1b;
    border: 1px solid #ff4444;
    border-radius: 8px;
    padding: 1rem;
    color: #ff8080;
}

.divider {
    border: none;
    border-top: 1px solid #2a3044;
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONFIG MODEL
# ─────────────────────────────────────────────────────────────
MODEL_DIR = "model"
GDRIVE_FOLDER_ID = "1QJVa0RN6E_lDaBqWrOaBfzFGJoE2mwf6"

def build_encoder():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128, 128, 1)),

        tf.keras.layers.Conv2D(
            32, (3, 3),
            activation="relu",
            padding="same"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(
            (2, 2),
            padding="same"
        ),

        tf.keras.layers.Conv2D(
            64, (3, 3),
            activation="relu",
            padding="same"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(
            (2, 2),
            padding="same"
        ),

        tf.keras.layers.Conv2D(
            128, (3, 3),
            activation="relu",
            padding="same"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(
            (2, 2),
            padding="same"
        ),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            128,
            activation="linear",
            name="latent_vector"
        )
    ])

    return model
# ─────────────────────────────────────────────────────────────
# DOWNLOAD + LOAD MODEL
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def download_and_load_models():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Download model jika belum ada
    if not os.path.exists(os.path.join(MODEL_DIR, "config_deploy.pkl")):
        with st.spinner("⏬ Mendownload model dari Google Drive (hanya sekali)..."):
            try:
                url = f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}"
                gdown.download_folder(
                    url,
                    output=MODEL_DIR,
                    quiet=False,
                    use_cookies=False
                )
            except Exception as e:
                return None, f"Gagal download model: {e}"

    # Cari folder model sebenarnya
    actual_dir = MODEL_DIR

    for item in os.listdir(MODEL_DIR):
        item_path = os.path.join(MODEL_DIR, item)

        if os.path.isdir(item_path):
            if os.path.exists(
                os.path.join(item_path, "config_deploy.pkl")
            ):
                actual_dir = item_path
                break

    try:
        # ===== LOAD MODEL KERAS =====
        encoder = build_encoder()
        
        encoder.load_weights(
        os.path.join(actual_dir, "model_encoder.h5")
        )

        # ===== LOAD PKL =====
        latent_scaler = joblib.load(
            os.path.join(actual_dir, "latent_scaler.pkl")
        )

        pca_img = joblib.load(
            os.path.join(actual_dir, "pca_img.pkl")
        )

        clinical_scaler = joblib.load(
            os.path.join(actual_dir, "clinical_scaler.pkl")
        )

        onehot = joblib.load(
            os.path.join(actual_dir, "onehot_category.pkl")
        )

        tfidf = joblib.load(
            os.path.join(actual_dir, "tfidf_vectorizer.pkl")
        )

        svd_text = joblib.load(
            os.path.join(actual_dir, "svd_text.pkl")
        )

        tfidf_scaler = joblib.load(
            os.path.join(actual_dir, "tfidf_scaler.pkl")
        )

        kmeans = joblib.load(
            os.path.join(actual_dir, "kmeans_model.pkl")
        )

        config = joblib.load(
            os.path.join(actual_dir, "config_deploy.pkl")
        )

        return (
            encoder,
            latent_scaler,
            pca_img,
            clinical_scaler,
            onehot,
            tfidf,
            svd_text,
            tfidf_scaler,
            kmeans,
            config
        ), None

    except Exception as e:
        return None, f"Gagal load model: {str(e)}"

# ─────────────────────────────────────────────────────────────
# TEXT PROCESSING
# ─────────────────────────────────────────────────────────────
NEGATION_PATTERNS = [
    r"tidak tampak",
    r"tidak terlihat",
    r"tidak ada",
    r"tanpa",
    r"tak tampak",
    r"no evidence of",
    r"negative for"
]


def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = text.replace("\n", " ")

    text = re.sub(
        r"[^a-zA-Z0-9\s\-\/]",
        " ",
        text
    )

    return re.sub(r"\s+", " ", text).strip()


def is_negated(text, keyword):

    for neg in NEGATION_PATTERNS:

        pattern = rf"{neg}\s+[\w\s]{{0,40}}{keyword}"

        if re.search(pattern, clean_text(text)):
            return True

    return False


def has_positive_keyword(text, keywords):

    text = clean_text(text)

    for kw in keywords:

        if re.search(kw, text):

            if not is_negated(text, kw):
                return 1

    return 0


def extract_report_features(text):

    t = clean_text(text)

    return pd.Series({
        "fitur_pneumonia":
            has_positive_keyword(
                t,
                [r"pneumonia", r"infiltrat",
                 r"konsolidasi", r"consolidation"]
            ),

        "fitur_bronchopneumonia":
            has_positive_keyword(
                t,
                [r"bronchopneumonia",
                 r"bronkopneumonia"]
            ),

        "fitur_fibrotic":
            has_positive_keyword(
                t,
                [r"fibrosis", r"fibrotic"]
            ),

        "fitur_efusi":
            has_positive_keyword(
                t,
                [r"efusi", r"effusion"]
            ),

        "fitur_bilateral":
            has_positive_keyword(
                t,
                [r"bilateral"]
            ),

        "fitur_pelebaran":
            has_positive_keyword(
                t,
                [r"kardiomegali",
                 r"cardiomegaly"]
            ),

        "fitur_normal":
            has_positive_keyword(
                t,
                [r"normal"]
            )
    })


CLINICAL_COLS = [
    "fitur_pneumonia",
    "fitur_bronchopneumonia",
    "fitur_fibrotic",
    "fitur_efusi",
    "fitur_bilateral",
    "fitur_pelebaran",
    "fitur_normal"
]

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown(
    '<p class="header-title">🫁 Pneumonia Clustering</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="header-sub">Sistem Klasifikasi Multimodal</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<hr class="divider">',
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────
result, err = download_and_load_models()

if result is None:
    st.markdown(
        f'<div class="error-box">❌ {err}</div>',
        unsafe_allow_html=True
    )
    st.stop()

(
    encoder,
    latent_scaler,
    pca_img,
    clinical_scaler,
    onehot,
    tfidf,
    svd_text,
    tfidf_scaler,
    kmeans,
    config
) = result

IMG_SIZE = config["img_size"]

st.success("✅ Model berhasil dimuat!")

# ─────────────────────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload X-Ray",
    type=["jpg", "jpeg", "png"]
)

report_text = st.text_area(
    "Laporan Radiologi"
)

if st.button("🔍 Analisis Cluster"):

    if uploaded_file is None:
        st.warning("Upload gambar dulu")
        st.stop()

    if not report_text.strip():
        st.warning("Isi laporan radiologi")
        st.stop()

    try:

        # ==================================================
        # PREPROCESS IMAGE
        # ==================================================
        img = Image.open(uploaded_file).convert("L")

        img = img.resize(
            (IMG_SIZE, IMG_SIZE)
        )

        arr = np.array(img).astype(
            "float32"
        ) / 255.0

        arr = np.expand_dims(
            arr,
            axis=-1
        )

        arr = np.expand_dims(
            arr,
            axis=0
        )

        # ==================================================
        # IMAGE FEATURE
        # ==================================================
        latent = encoder.predict(
            arr,
            verbose=0
        )

        latent_scaled = latent_scaler.transform(
            latent
        )

        latent_pca = pca_img.transform(
            latent_scaled
        )

        # ==================================================
        # CLINICAL FEATURE
        # ==================================================
        feat_series = extract_report_features(
            report_text
        )

        clinical_new = feat_series[
            CLINICAL_COLS
        ].astype(float).values.reshape(1, -1)

        clinical_scaled = clinical_scaler.transform(
            clinical_new
        )

        # ==================================================
        # CATEGORY
        # ==================================================
        category_text = "Normal"

        if feat_series["fitur_bronchopneumonia"] == 1:
            category_text = "Bronchopneumonia"

        elif feat_series["fitur_fibrotic"] == 1:
            category_text = "Fibrotic Pneumonia"

        elif feat_series["fitur_pneumonia"] == 1:
            category_text = "Pneumonia"

        category_onehot = onehot.transform(
            pd.DataFrame({
                "kategori_report": [category_text]
            })
        )

        # ==================================================
        # TFIDF FEATURE
        # ==================================================
        clean_report = clean_text(
            report_text
        )

        tfidf_new = tfidf.transform(
            [clean_report]
        )

        tfidf_svd = svd_text.transform(
            tfidf_new
        )

        tfidf_scaled = tfidf_scaler.transform(
            tfidf_svd
        )

        # ==================================================
        # GABUNG FITUR
        # ==================================================
        best_config = config["best_config"]

        iw = best_config["image_weight"]
        cw = best_config["clinical_weight"]
        catw = best_config["category_weight"]
        tw = best_config["tfidf_weight"]

        final_vector = np.hstack([

            latent_pca * iw,

            clinical_scaled * cw,

            category_onehot * catw,

            tfidf_scaled * tw

        ])

        # ==================================================
        # PREDIKSI CLUSTER
        # ==================================================
        cluster_pred = int(
            kmeans.predict(final_vector)[0]
        )

        cluster_interpretation = config[
            "cluster_interpretation"
        ]

        interpretasi = cluster_interpretation.get(
            cluster_pred,
            "Interpretasi belum tersedia"
        )

        # ==================================================
        # OUTPUT
        # ==================================================
       # ==================================================
# OUTPUT
# ==================================================
st.markdown(
    f"""
    <div class="result-card">

        <h3>🫁 Hasil Analisis</h3>

        <div class="cluster-badge">
            Cluster {cluster_pred}
        </div>

        <br>

        <p>
            <b>Kategori Report:</b><br>
            {category_text}
        </p>

        <p>
            <b>Interpretasi Penyakit:</b><br>
            {interpretasi}
        </p>

        <p>
            <b>Fitur Terdeteksi:</b>
        </p>

        <ul style="line-height:1.9; padding-left:20px;">
            <li>
                Pneumonia:
                {"✅" if feat_series["fitur_pneumonia"] else "❌"}
            </li>

            <li>
                Bronchopneumonia:
                {"✅" if feat_series["fitur_bronchopneumonia"] else "❌"}
            </li>

            <li>
                Fibrotic:
                {"✅" if feat_series["fitur_fibrotic"] else "❌"}
            </li>

            <li>
                Efusi:
                {"✅" if feat_series["fitur_efusi"] else "❌"}
            </li>

            <li>
                Bilateral:
                {"✅" if feat_series["fitur_bilateral"] else "❌"}
            </li>

            <li>
                Pelebaran:
                {"✅" if feat_series["fitur_pelebaran"] else "❌"}
            </li>

            <li>
                Normal:
                {"✅" if feat_series["fitur_normal"] else "❌"}
            </li>
        </ul>

    </div>
    """,
    unsafe_allow_html=True
)
