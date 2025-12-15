import streamlit as st
from PIL import Image
import os
from src.search_engine import SearchEngine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "data", "faiss_index", "index.bin")
METADATA_PATH = os.path.join(BASE_DIR, "data", "embeddings", "filenames.pkl")
IMAGE_DIR = os.path.join(BASE_DIR, "data", "images")

@st.cache_resource
def load_engine():
    return SearchEngine(INDEX_PATH, METADATA_PATH)

st.set_page_config(layout="wide", page_title="Neural Fashion Search")

# En-tête
st.title("🛍️ Neural Fashion Retrieval")
st.markdown("""
<style>
    .stButton>button { width: 100%; }
</style>
""", unsafe_allow_html=True)
st.info("Moteur de recherche hybride utilisant **OpenAI CLIP** + **FAISS**.")

try:
    engine = load_engine()
except Exception as e:
    st.error(f"Erreur au chargement du modèle : {e}")
    st.stop()

col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("🔍 Votre Recherche")
    search_mode = st.radio("Mode de recherche", ["Texte ✍️", "Image 🖼️"])
    
    results = []
    
    if search_mode == "Texte ✍️":
        query = st.text_input("Décrivez l'article :", "red summer dress")
        if st.button("Chercher") and query:
            with st.spinner("Recherche en cours..."):
                results = engine.search_text(query, k=15)
            
    elif search_mode == "Image 🖼️":
        uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Image requête", use_container_width=True)
            if st.button("Trouver similaires"):
                with st.spinner("Analyse visuelle..."):
                    results = engine.search_image(image, k=15)

with col2:
    st.subheader("🎯 Résultats")
    if results:
        cols = st.columns(5)
        for idx, (filename, score) in enumerate(results):
            with cols[idx % 5]:
                img_path = os.path.join(IMAGE_DIR, filename)
                if os.path.exists(img_path):
                    st.image(Image.open(img_path), use_container_width=True)
                    st.progress(min(int(score * 100), 100))
                    st.caption(f"Similaire à {int(score*100)}%")
                else:
                    st.warning(f"Image manquante: {filename}")
    elif results == [] and 'query' in locals():
        st.write("Aucun résultat trouvé.")