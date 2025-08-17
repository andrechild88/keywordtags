import streamlit as st
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import google.generativeai as genai
import json

# Konfigurasi API key Gemini
GOOGLE_API_KEY = "AIzaSyAo-QhLW7V9hTwy4cM63QqeVufzMr-33Sch8" 
genai.configure(api_key=GOOGLE_API_KEY)

# Prapemrosesan teks 
@st.cache_data
def preprocess_text(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    nltk.download('stopwords')
    list_stopwords = stopwords.words('indonesian')
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    filtered_words = [word for word in words if word not in list_stopwords]
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    return ' '.join(stemmed_words)

# MENGHASILKAN METADATA ---
def generate_metadata(artikel_baru, lokasi_berita="Banjarmasin"):
    artikel_clean = preprocess_text(artikel_baru)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf = tfidf_vectorizer.fit_transform([artikel_clean])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_scores = tfidf.toarray().flatten()

    keyword_df = pd.DataFrame({'keyword': feature_names, 'score': tfidf_scores})
    keyword_df = keyword_df.sort_values(by='score', ascending=False)
    top_keywords_list = keyword_df['keyword'].head(5).tolist()

    top_keywords_with_location = top_keywords_list + [lokasi_berita]
    keywords_for_prompt = ", ".join(top_keywords_with_location)

    model = genai.GenerativeModel('models/gemini-1.5-flash')
    
    prompt = f"""
    Berikut adalah artikel berita:
    "{artikel_baru}"

    Berikut adalah kata kunci utama dari artikel tersebut: {keywords_for_prompt}

    Tolong hasilkan metadata SEO yang optimal dalam format JSON berikut:
    {{
        "summary": "Ringkasan yang menarik dan informatif, maksimal 160 karakter",
        "keywords": ["5-7 kata kunci relevan dan spesifik"],
        "tags": ["5-7 tag, idealnya 1-2 kata, untuk mengelompokkan konten"]
    }}
    """
    response = model.generate_content(prompt)
    metadata_gemini = response.text
    metadata_json = json.loads(metadata_gemini.replace('```json\n', '').replace('\n```', ''))
    return metadata_json

# --- Tampilan UI Streamlit ---
st.title("Generator Metadata AI")
st.markdown("Masukkan artikel di bawah ini, lalu klik 'Generate' untuk mendapatkan metadata SEO.")

article_input = st.text_area("Tempel artikel di sini...", height=300)

if st.button("Generate Metadata"):
    if article_input:
        with st.spinner('Memproses...'):
            try:
                metadata = generate_metadata(article_input)
                st.subheader("Hasil Metadata")
                st.markdown("---")
                
                st.write("**Summary & Description:**")
                st.write(metadata.get('summary', ''))
                
                st.write("**Keywords:**")
                st.write(", ".join(metadata.get('keywords', [])))
                
                st.write("**Tags:**")
                st.write(", ".join(metadata.get('tags', [])))
                
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
    else:
        st.warning("Mohon masukkan konten artikel terlebih dahulu.")
