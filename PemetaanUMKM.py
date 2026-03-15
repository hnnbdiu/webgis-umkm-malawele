import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans # INI OTAK MACHINE LEARNING-NYA

# --- 1. KONFIGURASI TAMPILAN ---
st.set_page_config(page_title="GeoCluster Malawele | K-Means Spatial", layout="wide")

st.markdown("""
    <style>
    .header-box { background-color: var(--primary-color); padding: 1.2rem; border-radius: 8px; text-align: center; color: white; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-size: 2rem; font-weight: bold; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .sub-header { color: var(--primary-color); border-bottom: 2px solid var(--primary-color); padding-bottom: 0.5rem; margin-bottom: 1rem; font-weight: bold; font-size: 1.5rem; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD DATASET ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('dataset_umkm.csv', sep=None, engine='python')
        df.columns = df.columns.str.strip().str.lower()
        df = df.dropna(how='all')
        
        required_cols = ['nama', 'kategori', 'alamat', 'lat', 'lon']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            st.error(f"Sistem gagal menemukan kolom {missing}.")
            st.stop()
        
        # Bersihkan format lat/lon sejak awal
        df['lat'] = pd.to_numeric(df['lat'].astype(str).str.replace(',', '.'), errors='coerce')
        df['lon'] = pd.to_numeric(df['lon'].astype(str).str.replace(',', '.'), errors='coerce')
        df = df.dropna(subset=['lat', 'lon']) # Buang data cacat
            
        return df
    except FileNotFoundError:
        st.error("FATAL ERROR: File 'dataset_umkm.csv' tidak ditemukan.")
        st.stop()

df = load_data()

# --- 3. ANTARMUKA PENCARIAN (UI) ---
st.markdown('<div class="header-box"> WebGIS K-Means Clustering UMKM Malawele</div>', unsafe_allow_html=True)
search_query = st.text_input("", placeholder="Ketik Nama Jalan atau Nama UMKM...")

if not df.empty:
    if search_query:
        mask = df['alamat'].astype(str).str.contains(search_query, case=False, na=False) | \
               df['nama'].astype(str).str.contains(search_query, case=False, na=False)
        filtered_df = df[mask].copy()
    else:
        filtered_df = df.copy()

    # --- 4. EKSEKUSI MACHINE LEARNING (K-MEANS) ---
    # Kita bagi menjadi 3 zona klaster (syarat: minimal harus ada 3 data UMKM)
    if len(filtered_df) >= 3:
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        filtered_df['cluster'] = kmeans.fit_predict(filtered_df[['lat', 'lon']])
    else:
        # Jika data kurang dari 3 hasil pencarian, paksa masuk ke klaster 0
        filtered_df['cluster'] = 0

    # --- 5. RENDER PETA ---
    col_map, col_list = st.columns([2, 1])
    
    with col_map:
        # PUSAT PETA MUTLAK KELURAHAN MALAWELE
        center_lat = -0.9648
        center_lon = 131.3059
        m = folium.Map(location=[center_lat, center_lon], zoom_start=15)
        
        # Palet warna untuk membedakan hasil zona K-Means
        warna_klaster = {0: 'red', 1: 'blue', 2: 'green'}
        
        for _, row in filtered_df.iterrows():
            lat, lon = row['lat'], row['lon']
            zona_id = row['cluster']
            warna_pin = warna_klaster.get(zona_id, 'gray')
            
            google_maps_url = f"https://www.google.com/maps/dir/?api=1&destination={lat},{lon}"
            popup_html = f"""
            <div style="font-family: Arial; font-size: 12px;">
                <b>{row['nama']}</b><br>
                Kategori: {row['kategori']}<br>
                {row['alamat']}<br>
                <b>Zona Klaster: {zona_id + 1}</b><br><br>
                <a href="{google_maps_url}" target="_blank" style="background-color: #28a745; color: white; padding: 6px 12px; text-align: center; text-decoration: none; display: inline-block; border-radius: 4px; font-weight: bold;">
                    🗺️ Rute ke Sini
                </a>
            </div>
            """
            
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"Klik untuk detail (Zona {zona_id + 1})",
                icon=folium.Icon(color=warna_pin, icon="info-sign")
            ).add_to(m)
                
        st_folium(m, width="100%", height=500)

    with col_list:
        st.markdown('<div class="sub-header">📑 Direktori UMKM</div>', unsafe_allow_html=True)
        if not filtered_df.empty:
            # Tampilkan juga kolom zona di tabel agar terlihat ilmiah
            display_df = filtered_df[['nama', 'kategori', 'cluster']].copy()
            display_df.columns = ['Nama UMKM', 'Kategori', 'Zona Klaster']
            display_df['Zona Klaster'] = display_df['Zona Klaster'] + 1
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Data tidak ditemukan.")