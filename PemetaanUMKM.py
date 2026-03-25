import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
import google.generativeai as genai
import re

# --- 1. KONFIGURASI AI ---
# Masukkan API Key Anda di sini jika ingin fitur AI-nya hidup.
# Jika dibiarkan, sistem akan otomatis murni memakai pencarian lokal.
API_KEY = "MASUKKAN_API_KEY_ANDA_DI_SINI" 

if API_KEY != "MASUKKAN_API_KEY_ANDA_DI_SINI":
    try:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception:
        model = None
else:
    model = None

st.set_page_config(page_title="WebGIS UMKM | Hybrid Engine", layout="wide")

# --- 2. LOAD DATA (ANTI-ERROR PARSER) ---
@st.cache_data
def load_data():
    try:
        # Parameter on_bad_lines='skip' akan membuang baris cacat secara otomatis
        df = pd.read_csv('dataset_umkm.csv', sep=',', engine='python', on_bad_lines='skip')
        df.columns = df.columns.str.strip().str.lower()
        
        df['lat'] = pd.to_numeric(df['lat'].astype(str).str.replace(',', '.'), errors='coerce')
        df['lon'] = pd.to_numeric(df['lon'].astype(str).str.replace(',', '.'), errors='coerce')
        df = df.dropna(subset=['lat', 'lon', 'nama'])
        
        # Klastering K-Means
        if len(df) >= 3:
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            df['cluster'] = kmeans.fit_predict(df[['lat', 'lon']])
        else:
            df['cluster'] = 0
            
        return df
    except Exception as e:
        st.error(f"Eror Kritis Dataset: {e}")
        st.stop()

df = load_data()

# --- 3. LOGIKA MESIN (AI & MANUAL) ---
def panggil_ai(query, data_konteks):
    if not model: 
        return None
    try:
        prompt = f"""
        Anda adalah Asisten UMKM Malawele. Jawab pertanyaan user: '{query}'.
        Gunakan data referensi ini (jika relevan): {data_konteks.to_dict()}
        Jawab dengan presisi, singkat, dan jangan mengarang data.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception:
        return None

def smart_filter(query, data):
    if not query: 
        return data
        
    stopwords = ['di', 'dan', 'yang', 'ada', 'ke', 'dari', 'pada']
    keywords = [k for k in re.findall(r'\w+', query.lower()) if k not in stopwords]
    
    if not keywords: 
        return data

    def logic(row):
        text = f"{row['nama']} {row['alamat']} {row.get('kategori', '')}".lower()
        text = re.sub(r'[^\w\s]', '', text) 
        return all(k in text for k in keywords)
        
    return data[data.apply(logic, axis=1)]

# --- 4. ANTARMUKA PENGGUNA (UI) ---
st.markdown("### 📍 WebGIS UMKM Malawele (Hybrid AI)")
query_user = st.text_input("🔍 Cari Lokasi atau Tanya AI:", placeholder="Contoh: bengkel di jl wortel")

filtered_df = smart_filter(query_user, df)

col_map, col_ai = st.columns([2, 1])

# Render Peta
with col_map:
    if not filtered_df.empty:
        start_loc = [filtered_df['lat'].mean(), filtered_df['lon'].mean()]
        zoom = 17 if len(filtered_df) < 5 else 16
    else:
        # Default koordinat Sorong/Aimas
        start_loc = [-0.9648, 131.3059] 
        zoom = 15

    m = folium.Map(location=start_loc, zoom_start=zoom)
    colors = {0: 'red', 1: 'blue', 2: 'green'}
    
    for _, row in filtered_df.iterrows():
        popup_html = f"""
        <div style="font-family: sans-serif; min-width: 160px;">
            <b style="color: #1f1f1f;">{row['nama'].upper()}</b><br>
            <span style="font-size: 12px; color: #555;">{row['alamat']}</span>
            <hr style="margin: 8px 0; border: 0; border-top: 1px solid #ddd;">
            <a href="https://www.google.com/maps/dir/?api=1&destination={row['lat']},{row['lon']}" 
               target="_blank" 
               style="background-color: #4285F4; color: white; padding: 8px 10px; text-decoration: none; border-radius: 4px; font-size: 11px; display: block; text-align: center; font-weight: bold;">
               🚗 BUKA RUTE GOOGLE MAPS
            </a>
        </div>
        """
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=folium.Popup(popup_html, max_width=250),
            icon=folium.Icon(color=colors.get(row['cluster'], 'gray'), icon='info-sign')
        ).add_to(m)
        
    st_folium(m, width="100%", height=500, key="map")

# Render AI / Hasil Manual
with col_ai:
    st.subheader("🤖 Analisis Sistem")
    
    if query_user:
        with st.spinner("Memproses kueri..."):
            jawaban_ai = panggil_ai(query_user, filtered_df.head(5))
            
            if jawaban_ai:
                st.success("⚡ AI Respons:")
                st.write(jawaban_ai)
                st.caption("Peta di samping juga telah disaring berdasarkan kueri Anda.")
            else:
                st.warning("⚠️ AI Offline / Limit. Mode Pencarian Lokal Aktif:")
                if filtered_df.empty:
                    st.write("Tidak ada kecocokan data.")
                else:
                    st.dataframe(filtered_df[['nama', 'alamat']], hide_index=True)
    else:
        st.info(f"Sistem siap. Total data valid dirender: {len(df)} UMKM.")
