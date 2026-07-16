import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
import google.generativeai as genai
import re
import urllib.parse
import os
from streamlit_geolocation import streamlit_geolocation

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="GIS UMKM | Malawele",
    layout="wide",
    initial_sidebar_state="auto",
    page_icon="📍"
)

if 'last_query' not in st.session_state:
    st.session_state.last_query = ""
if 'ai_response' not in st.session_state:
    st.session_state.ai_response = ""
if 'preprocessed_query' not in st.session_state:
    st.session_state.preprocessed_query = ""

# ==========================================
# INJEKSI CSS TEMA BIRU TUA - GIS PROFESIONAL
# ==========================================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Exo+2:wght@300;400;600;700;800&family=Share+Tech+Mono&family=Nunito:wght@300;400;600&display=swap" rel="stylesheet">

<style>
/* ===== ROOT & VARIABEL WARNA ===== */
:root {
    --navy-deepest:  #060E1F;
    --navy-dark:     #0B1A33;
    --navy-mid:      #112340;
    --navy-card:     #0F1F3D;
    --navy-border:   #1A3560;
    --navy-hover:    #1E3E6E;
    --cyan-bright:   #00C8FF;
    --cyan-soft:     #38BDF8;
    --gold:          #F0A500;
    --gold-soft:     #FCD34D;
    --white:         #E8F0FE;
    --text-muted:    #7FA0C8;
    --success:       #10B981;
    --warning:       #F59E0B;
    --danger:        #EF4444;
    --glow-cyan:     0 0 18px rgba(0,200,255,0.35);
    --glow-gold:     0 0 18px rgba(240,165,0,0.3);
}

/* Bungkus komponen GPS jadi kotak kecil bertema navy. Wrapper INI adalah
   jaring pengaman: background-nya solid navy sesuai desain, jadi APAPUN
   sisa area yang tidak tertutup oleh iframe di dalamnya akan tetap
   terlihat navy — bukan hitam/putih kosong. flex + center di sini yang
   BERTUGAS memusatkan ikonnya secara horizontal & vertikal. */
div[data-testid="stCustomComponentV1"],
[data-testid="stElementContainer"]:has(iframe[title*="geolocation"]) {
    background: linear-gradient(160deg, #0D2040 0%, #0A1628 100%) !important;
    border: 1px solid var(--navy-border) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 48px !important;
    height: 48px !important;
    max-width: 48px !important;
    max-height: 48px !important;
    box-shadow: var(--glow-cyan);
}

/* Iframe DIKEMBALIKAN ke ukuran alaminya (bukan di-stretch 100%) supaya
   layout internal tombolnya tidak terdistorsi/nempel pojok. Karena wrapper
   di atas sudah flex+center, iframe seukuran ini otomatis diposisikan
   TEPAT DI TENGAH kotak 48x48 oleh browser — bukan lagi nempel kiri-atas. */
iframe[title*="geolocation"] {
    background-color: transparent !important;
    color-scheme: dark !important;
    border: none !important;
    border-radius: 8px !important;
    width: 44px !important;
    height: 44px !important;
    flex-shrink: 0 !important;
}




header[data-testid="stHeader"],
header[data-testid="stHeader"] * {
    background-color: #060E1F !important;
    border-bottom: 1px solid #1A3560 !important;
}

[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stVerticalBlock"],
[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stBottom"],
section[data-testid="stSidebar"] > div,
.stMainBlockContainer,
.main .block-container,
div.stApp, div[class*="appview"],
div[class*="main"], div[class*="block"] {
    background-color: transparent !important;
    color: var(--white) !important;
}

/* Tutup background putih di toolbar atas */
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {
    background-color: #060E1F !important;
}

/* Paksa semua teks tetap terang */
*, *::before, *::after {
    color: inherit;
}

p, span, div, li, td, th, label, caption {
    color: var(--white);
}

.stApp {
    background-color: var(--navy-deepest);
    background-image:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(0,100,200,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(0,200,255,0.07) 0%, transparent 55%),
        linear-gradient(180deg, var(--navy-deepest) 0%, #0A1628 100%);
    background-attachment: fixed;
}

/* Grid pattern overlay */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,150,255,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,150,255,0.04) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* ===== FONT GLOBAL ===== */
html, body, p, label {
    font-family: 'Nunito', sans-serif !important;
    color: var(--white);
}

/* Jangan timpa font Material Icons milik Streamlit */
[class*="css"]:not([class*="material"]):not([data-testid="collapsedControl"]) {
    font-family: 'Nunito', sans-serif;
    color: var(--white);
}

/* Pulihkan font ikon Material secara eksplisit */
.material-icons,
[data-testid="collapsedControl"],
[data-testid="collapsedControl"] *,
button[kind="header"],
button[kind="header"] * {
    font-family: 'Material Icons' !important;
    color: var(--white) !important;
}

h1, h2, h3, h4 {
    font-family: 'Exo 2', sans-serif !important;
    letter-spacing: 0.04em;
}

/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #071426 0%, #0B1E3B 50%, #0A1628 100%) !important;
    border-right: 1px solid var(--navy-border) !important;
    box-shadow: 4px 0 24px rgba(0,0,0,0.5);
}

[data-testid="stSidebar"] * {
    color: var(--white) !important;
}

[data-testid="stSidebar"] .stRadio label {
    font-family: 'Exo 2', sans-serif !important;
    font-size: 0.9rem;
    padding: 6px 0;
    transition: color 0.2s;
}

[data-testid="stSidebar"] .stRadio label:hover {
    color: var(--cyan-bright) !important;
}

[data-testid="stSidebar"] hr {
    border-color: var(--navy-border) !important;
}

[data-testid="stSidebar"] .stCaption {
    color: var(--text-muted) !important;
    font-size: 0.75rem;
}

/* ===== HEADER HERO ===== */
.hero-header {
    background: linear-gradient(135deg, #0D2448 0%, #0A1A35 60%, #071224 100%);
    border: 1px solid var(--navy-border);
    border-left: 4px solid var(--cyan-bright);
    border-radius: 12px;
    padding: 20px 28px 16px 28px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
    box-shadow: var(--glow-cyan), 0 8px 32px rgba(0,0,0,0.4);
}

.hero-header::after {
    content: '';
    position: absolute;
    top: -30px; right: -30px;
    width: 160px; height: 160px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0,200,255,0.08), transparent 70%);
}

.hero-header h1 {
    font-family: 'Exo 2', sans-serif !important;
    font-size: 1.65rem;
    font-weight: 800;
    color: var(--white) !important;
    margin: 0 0 4px 0;
    letter-spacing: 0.05em;
    text-shadow: 0 0 20px rgba(0,200,255,0.4);
}

.hero-header .hero-sub {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.72rem;
    color: var(--cyan-bright) !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    opacity: 0.85;
}

.hero-status {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16,185,129,0.12);
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.7rem;
    color: #10B981 !important;
    font-family: 'Share Tech Mono', monospace !important;
    letter-spacing: 0.08em;
}

.dot-pulse {
    width: 7px; height: 7px;
    background: #10B981;
    border-radius: 50%;
    animation: pulse 2s ease-in-out infinite;
    display: inline-block;
}

@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(16,185,129,0.6); }
    50%       { opacity: 0.6; box-shadow: 0 0 0 5px rgba(16,185,129,0); }
}

/* ===== KARTU KONTROL PENCARIAN ===== */
.control-card {
    background: linear-gradient(135deg, var(--navy-card), #0D1E3A);
    border: 1px solid var(--navy-border);
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

/* ===== TOMBOL CARI (form_submit_button) ===== */
[data-testid="stFormSubmitButton"] button,
.stButton button {
    background: linear-gradient(135deg, var(--cyan-bright), #0072B5) !important;
    color: #06131F !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 800 !important;
    box-shadow: var(--glow-cyan) !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
[data-testid="stFormSubmitButton"] button:hover,
.stButton button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 0 22px rgba(0,200,255,0.55) !important;
}
[data-testid="stForm"] {
    border: none !important;
    padding: 0 !important;
}

/* ===== INPUT FIELDS ===== */
.stTextInput > div > div > input {
    background: #0A1628 !important;
    border: 1px solid var(--navy-border) !important;
    border-radius: 8px !important;
    color: var(--white) !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 10px 14px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--cyan-bright) !important;
    box-shadow: var(--glow-cyan) !important;
    outline: none !important;
}

.stTextInput > div > div > input::placeholder {
    color: var(--text-muted) !important;
    font-style: italic;
}

/* ===== LABEL WIDGET ===== */
.stTextInput label,
.stSelectbox label,
.stRadio label,
label[data-testid="stWidgetLabel"] {
    font-family: 'Exo 2', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: var(--text-muted) !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* ===== SELECTBOX ===== */
.stSelectbox > div > div {
    background: #0A1628 !important;
    border: 1px solid var(--navy-border) !important;
    border-radius: 8px !important;
    color: var(--white) !important;
}

.stSelectbox > div > div:hover {
    border-color: var(--cyan-soft) !important;
}

.stSelectbox > div > div > div {
    color: var(--white) !important;
}

/* ===== RADIO BUTTONS ===== */
.stRadio > div {
    gap: 8px;
}

.stRadio > div > label {
    background: #0A1628 !important;
    border: 1px solid var(--navy-border) !important;
    border-radius: 6px !important;
    padding: 6px 14px !important;
    font-size: 0.82rem !important;
    transition: all 0.2s !important;
    cursor: pointer;
    text-transform: none !important;
    letter-spacing: 0 !important;
    color: var(--white) !important;
}

.stRadio > div > label:hover {
    border-color: var(--cyan-bright) !important;
    background: var(--navy-hover) !important;
}

/* ===== DIVIDER ===== */
hr {
    border: none !important;
    border-top: 1px solid var(--navy-border) !important;
    margin: 16px 0 !important;
}

/* ===== PETA CONTAINER ===== */
.map-wrapper {
    background: var(--navy-card);
    border: 1px solid var(--navy-border);
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 20px;
    box-shadow: var(--glow-cyan), 0 8px 30px rgba(0,0,0,0.5);
}

.map-title {
    font-family: 'Exo 2', sans-serif !important;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--cyan-bright) !important;
    padding: 10px 16px 6px;
    border-bottom: 1px solid var(--navy-border);
    background: linear-gradient(90deg, rgba(0,200,255,0.06), transparent);
}

/* ===== PANEL AI ===== */
.ai-panel {
    background: linear-gradient(160deg, #0D2040 0%, #0A1628 100%);
    border: 1px solid var(--navy-border);
    border-top: 3px solid var(--gold);
    border-radius: 12px;
    padding: 18px;
    height: auto;
    margin-bottom: 20px;
    box-shadow: var(--glow-gold), 0 6px 24px rgba(0,0,0,0.4);
}

.ai-title {
    font-family: 'Exo 2', sans-serif !important;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--gold) !important;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 14px;
}

.ai-response-box {
    background: rgba(0,200,255,0.05);
    border: 1px solid rgba(0,200,255,0.2);
    border-left: 3px solid var(--cyan-bright);
    border-radius: 8px;
    padding: 14px 16px;
    font-size: 0.88rem;
    line-height: 1.65;
    color: var(--white) !important;
    font-family: 'Nunito', sans-serif !important;
}

.standby-box {
    background: rgba(17,35,64,0.6);
    border: 1px dashed var(--navy-border);
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    color: var(--text-muted) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.8rem;
    letter-spacing: 0.1em;
}

/* ===== ALERT / NOTIFICATION BOXES ===== */
.stAlert {
    border-radius: 8px !important;
    border: none !important;
}

div[data-baseweb="notification"] {
    background: rgba(16,185,129,0.1) !important;
    border: 1px solid rgba(16,185,129,0.3) !important;
    border-radius: 8px !important;
}

div[data-baseweb="notification"] * {
    color: #6EE7B7 !important;
}

/* Warning */
.stAlert [data-testid="stNotificationContentWarning"] {
    background: rgba(245,158,11,0.1) !important;
    border: 1px solid rgba(245,158,11,0.3) !important;
}

/* Error */
.stAlert [data-testid="stNotificationContentError"] {
    background: rgba(239,68,68,0.1) !important;
    border: 1px solid rgba(239,68,68,0.3) !important;
}

/* Info */
[data-testid="stNotificationContentInfo"],
.stInfo > div {
    background: rgba(0,100,200,0.1) !important;
    border: 1px solid rgba(0,150,255,0.25) !important;
    border-radius: 8px !important;
    color: var(--cyan-soft) !important;
}

/* ===== DATAFRAME ===== */
[data-testid="stDataFrame"] {
    border: 1px solid var(--navy-border) !important;
    border-radius: 8px !important;
    overflow: hidden;
}

[data-testid="stDataFrame"] th {
    background: #0D2040 !important;
    color: var(--cyan-bright) !important;
    font-family: 'Exo 2', sans-serif !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid var(--navy-border) !important;
}

[data-testid="stDataFrame"] td {
    background: var(--navy-card) !important;
    color: var(--white) !important;
    font-size: 0.85rem !important;
    border-bottom: 1px solid rgba(26,53,96,0.5) !important;
}

[data-testid="stDataFrame"] tr:hover td {
    background: var(--navy-hover) !important;
}

/* ===== SPINNER ===== */
.stSpinner > div {
    border-top-color: var(--cyan-bright) !important;
}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--navy-dark); }
::-webkit-scrollbar-thumb { background: var(--navy-border); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--cyan-bright); }

/* ===== TENTANG PAGE ===== */
.about-card {
    background: linear-gradient(135deg, var(--navy-card), #0A1B32);
    border: 1px solid var(--navy-border);
    border-radius: 12px;
    padding: 24px 28px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.35);
}

.about-card h3 {
    font-family: 'Exo 2', sans-serif !important;
    font-size: 1rem;
    font-weight: 700;
    color: var(--cyan-bright) !important;
    letter-spacing: 0.06em;
    margin-top: 20px;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(0,200,255,0.15);
}

.about-card h3:first-child { margin-top: 0; }

.about-card p, .about-card li {
    font-size: 0.88rem;
    line-height: 1.7;
    color: #B0C8E8 !important;
}

.about-card strong {
    color: var(--gold-soft) !important;
}

.badge-tech {
    display: inline-block;
    background: rgba(0,200,255,0.1);
    border: 1px solid rgba(0,200,255,0.25);
    color: var(--cyan-soft) !important;
    font-size: 0.72rem;
    font-family: 'Share Tech Mono', monospace !important;
    padding: 2px 10px;
    border-radius: 4px;
    margin: 2px;
    letter-spacing: 0.05em;
}

.member-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 12px;
    background: rgba(17,35,64,0.5);
    border: 1px solid var(--navy-border);
    border-radius: 8px;
    margin-bottom: 8px;
}

.member-num {
    background: var(--navy-border);
    color: var(--cyan-bright) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.75rem;
    width: 26px; height: 26px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}

/* ===== CAPTION ===== */
.stCaption, .caption {
    color: var(--text-muted) !important;
    font-size: 0.78rem;
}

/* ===== IMAGE SIDEBAR ===== */
[data-testid="stSidebar"] img {
    filter: drop-shadow(0 0 12px rgba(0,200,255,0.4));
}

/* Sembunyikan tombol fullscreen di semua gambar & peta */
[data-testid="StyledFullScreenButton"],
[data-testid="stFullScreenFrame"] > button,
.stIFrame ~ button,
button[title="View fullscreen"] {
    display: none !important;
}

/* ===== SIDEBAR TITLE ===== */
[data-testid="stSidebar"] h3 {
    font-family: 'Exo 2', sans-serif !important;
    font-size: 0.85rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--cyan-bright) !important;
}

/* ===== FOLIUM MAP BORDER ===== */
iframe {
    border-radius: 0 0 10px 10px !important;
    border: none !important;
}

/* ===== BLOCK CONTAINER PADDING ===== */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* =============================================
   RESPONSIF — TABLET (max 1024px)
   ============================================= */
@media (max-width: 1024px) {
    .hero-header h1 {
        font-size: 1.3rem !important;
    }
    .block-container {
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
    }
    .ai-panel {
        padding: 14px !important;
    }
}

/* =============================================
   RESPONSIF — MOBILE (max 768px)
   ============================================= */
@media (max-width: 768px) {

    /* Padding konten lebih rapat */
    .block-container {
        padding-top: 0.8rem !important;
        padding-left: 0.7rem !important;
        padding-right: 0.7rem !important;
        padding-bottom: 1.2rem !important;
    }

    /* Hero header lebih kompak */
    .hero-header {
        padding: 14px 16px 12px 16px !important;
        margin-bottom: 12px !important;
        border-left-width: 3px !important;
    }
    .hero-header h1 {
        font-size: 1.05rem !important;
        letter-spacing: 0.03em !important;
    }
    .hero-header .hero-sub {
        font-size: 0.62rem !important;
        letter-spacing: 0.08em !important;
    }
    .hero-status {
        font-size: 0.62rem !important;
        padding: 2px 8px !important;
    }

    /* Control card lebih rapat */
    .control-card {
        padding: 12px 14px !important;
        margin-bottom: 10px !important;
    }

    /* Input dan selectbox full width */
    .stTextInput, .stSelectbox {
        width: 100% !important;
    }
    .stTextInput > div > div > input {
        font-size: 1rem !important;
        padding: 10px 12px !important;
    }

    /* Radio tombol lebih kecil */
    .stRadio > div > label {
        padding: 5px 10px !important;
        font-size: 0.78rem !important;
    }

    /* Peta lebih pendek di HP (khusus iframe di dalam .map-wrapper,
       supaya TIDAK ikut menimpa iframe komponen GPS) */
    .map-wrapper iframe {
        height: 320px !important;
    }

    /* AI panel full width, margin atas */
    .ai-panel {
        padding: 14px !important;
        margin-top: 10px !important;
        height: auto !important;
    }
    .ai-title {
        font-size: 0.78rem !important;
    }
    .ai-response-box {
        font-size: 0.83rem !important;
        padding: 11px 13px !important;
    }
    .standby-box {
        padding: 16px !important;
        font-size: 0.72rem !important;
    }

    /* Map title */
    .map-title {
        font-size: 0.7rem !important;
        padding: 8px 12px 5px !important;
    }

    /* Dataframe scroll horizontal */
    [data-testid="stDataFrame"] {
        overflow-x: auto !important;
        font-size: 0.8rem !important;
    }

    /* About card */
    .about-card {
        padding: 16px 18px !important;
    }
    .about-card h3 {
        font-size: 0.9rem !important;
    }
    .about-card p, .about-card li {
        font-size: 0.82rem !important;
    }

    /* Member item */
    .member-item {
        padding: 7px 10px !important;
    }

    /* Badge tech */
    .badge-tech {
        font-size: 0.65rem !important;
        padding: 2px 7px !important;
    }

    /* Sidebar auto-collapse hint */
    [data-testid="stSidebar"] {
        min-width: 220px !important;
        max-width: 260px !important;
    }

    /* Divider margin */
    hr {
        margin: 10px 0 !important;
    }

    /* ============================================
       PAKSA SEMUA st.columns() TUMPUK VERTIKAL
       (root cause layout mobile tidak responsif)
       ============================================ */
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        flex-wrap: wrap !important;
        gap: 0 !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
    [data-testid="stHorizontalBlock"] > [data-testid="column"],
    [data-testid="stHorizontalBlock"] > div[class*="column"] {
        width: 100% !important;
        min-width: 100% !important;
        max-width: 100% !important;
        flex: 1 1 100% !important;
        margin-bottom: 14px !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child,
    [data-testid="stHorizontalBlock"] > [data-testid="column"]:last-child {
        margin-bottom: 0 !important;
    }

    /* Radio "Metode Lokasi" jadi horizontal rapi (bukan vertikal tumpuk aneh) */
    .stRadio > div {
        flex-direction: row !important;
        flex-wrap: wrap !important;
    }

    /* Peta & AI panel tidak lagi berdempetan tanpa jarak */
    .map-wrapper {
        margin-bottom: 16px !important;
    }
}

/* =============================================
   RESPONSIF — HP KECIL (max 480px)
   ============================================= */
@media (max-width: 480px) {
    .hero-header h1 {
        font-size: 0.95rem !important;
    }
    .hero-header .hero-sub {
        display: none !important;
    }
    .hero-status {
        margin-top: 6px !important;
    }
    .block-container {
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
    }
    .map-wrapper iframe {
        height: 280px !important;
    }

    /* Sidebar full-width saat dibuka di HP kecil */
    [data-testid="stSidebar"] {
        min-width: 100% !important;
        max-width: 100% !important;
    }

    /* Kolom tetap tumpuk vertikal di breakpoint ini juga */
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
    [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        width: 100% !important;
        min-width: 100% !important;
        max-width: 100% !important;
    }

    /* Konsol pencarian & filter kategori: teks input jangan sampai zoom otomatis di iOS */
    .stTextInput > div > div > input,
    .stSelectbox > div > div {
        font-size: 16px !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# MENU NAVIGASI (SIDEBAR)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/854/854878.png", width=90)
    st.markdown("### 🗺 Menu Navigasi")
    menu_pilihan = st.radio("Pilih Halaman:", ["📍 Peta UMKM (Utama)", "ℹ️ Tentang Web"])
    st.divider()
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace; font-size:0.7rem; color:#4A7AB5; line-height:1.8;">
    SYS  : GIS UMKM v2.0<br>
    LOC  : Malawele, Sorong<br>
    MODE : NLP + RAG Hybrid<br>
    YEAR : 2026
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.caption("© 2026 | Kelompok 14 — Teknik Informatika UM Sorong")

# --- API & MODE SISTEM ---
# Sistem akan mengambil API Key secara aman dari file .streamlit/secrets.toml
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except (KeyError, FileNotFoundError):
    API_KEY = ""

if API_KEY.strip() != "":
    try:
        genai.configure(api_key=API_KEY)
        mode_sistem = "LLM"
    except Exception:
        mode_sistem = "Pakar"
else:
    mode_sistem = "Pakar"

# --- FUNGSI MATEMATIKA SPASIAL ---
def hitung_jarak_vektor(lat1, lon1, lat2_series, lon2_series):
    R = 6371.0
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2_series.values), np.radians(lon2_series.values)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# --- PIPELINE DATA ---
@st.cache_data(ttl=3600)
def load_data():
    try:
        df = pd.read_csv('dataset_umkm.csv', sep=',', engine='python', on_bad_lines='skip')
        df.columns = df.columns.str.strip().str.lower()
        df['lat'] = pd.to_numeric(df['lat'].astype(str).str.replace(',', '.'), errors='coerce')
        df['lon'] = pd.to_numeric(df['lon'].astype(str).str.replace(',', '.'), errors='coerce')
        df = df.dropna(subset=['lat', 'lon', 'nama']).copy()
        if len(df) >= 3:
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            df['cluster'] = kmeans.fit_predict(df[['lat', 'lon']])
        else:
            df['cluster'] = 0
        return df
    except FileNotFoundError:
        st.error("Kritis: File 'dataset_umkm.csv' tidak ditemukan.")
        st.stop()

df = load_data()

# --- ENGINE NLP ---
@st.cache_resource
def inisialisasi_nlp():
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
        stemmer = StemmerFactory().create_stemmer()
        stopwords_dasar = StopWordRemoverFactory().get_stop_words()
        custom_slang = [
            'info', 'infokan', 'inpo', 'spill', 'tolong', 'kasih', 'tau', 'aja', 'saja', 'kasi', 'dimana', 'kah', 'kh', 'hem', 'dulu', 'mas', 'cari', 'mencari', 'sedang',
            'banget', 'bgt', 'paling', 'sekali', 'skali', 'coba', 'bisa', 'dong', 'sih', 'rek', 'rekomendasi', 'cariin', 'z', 'dongs','we','pos','nongki','biyasa','km','ddk','bru','jln','p','nobar',
            'ko', 'sa', 'tra', 'trada', 'tara', 'su', 'pi', 'pu', 'tong', 'kam', 'de', 'umkm', 'skli', 'inpokan', 'inpoin', 'infoin','sa', 'ka', 'mo', 'ke', 'bos', 'poster',         
            'dolo', 'dlu', 'toh', 'ka', 'kah', 'pace', 'mace', 'kaka', 'mo', 't4', 'carikan', 'cari', 'dmn', 'dimana', 'ya', 'yah', 'nyak',
            'yg', 'utk', 'kalo', 'klo', 'udah', 'udh', 'bs', 'bsa', 'gmn', 'gimana', 'pas', 'mo', 'mau','saya', 'aku', 'sy', 'plis', 'hemz', 'kaks', 'bro', 'kiw',
            'mkn', 'makan', 'mabar', 'skuy', 'kuy', 'pe', 'tempat', 'bah', 'ah', 'we', 'kawan', 'teman', 'aban', 'abang', 'bang', 'bg', 'ngab', 'cik', 'mintol', 'minta tolong'
        ]
        kata_spasial_lindungi = {'dekat', 'sekitar', 'sini', 'area', 'wilayah'}
        stopwords_final = set(stopwords_dasar + custom_slang) - kata_spasial_lindungi
        return stemmer, stopwords_final
    except ImportError:
        st.error("Kritis: Library 'Sastrawi' belum diinstal. Jalankan: pip install Sastrawi")
        st.stop()

# --- PRE-PROCESSING LLM (GEMINI) UNTUK DIALEK LOKAL & TYPO ---
def preprocess_query_with_llm(raw_query, mode):
    if mode != "LLM" or not raw_query.strip():
        return raw_query # Fallback ke Sastrawi jika API tidak aktif
    
    prompt = f"""
    Anda adalah asisten pemrosesan bahasa alami untuk sistem pencarian spasial di Sorong, Papua Barat Daya.
    Tugas utama Anda adalah:
    1. Memperbaiki kesalahan ketik (typo) pada kueri pengguna.
    2. Menerjemahkan kueri percakapan (termasuk dialek lokal Papua/Sorong) menjadi KATA KUNCI BENDA/OBJEK baku dalam bahasa Indonesia yang siap dicari di database.
    
    ATURAN KETAT:
    1. Hapus semua kata sapaan, subjek (saya, sa, ko, dll), kata kerja (makan, pergi, pi, dll), dan kata keterangan.
    2. Pertahankan kata yang berhubungan dengan lokasi spasial JIKA ADA (dekat, sekitar, sini).
    3. Perbaiki ejaan yang salah. Contoh: "wrg" menjadi "warung", "bngkel" menjadi "bengkel", "mkn" menjadi "makan", "angkringn" menjadi "angkringan".
    4. Hasilkan HANYA kata kuncinya saja, tanpa penjelasan, tanpa tanda kutip.
    
    Contoh 1:
    Kueri Mentah: "sa mau mkn di warunk dolo"
    Output: warung
    
    Contoh 2:
    Kueri Mentah: "infokan bgkel motor dkt sni kah"
    Output: bengkel motor dekat sini
    
    Contoh 3:
    Kueri Mentah: "ad jual alat tulis kh"
    Output: fotokopi alat tulis
    
    Sekarang, terjemahkan dan perbaiki kueri ini:
    Kueri Mentah: "{raw_query}"
    Output:
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1, # Suhu rendah untuk memastikan determinisme koreksi kata
                max_output_tokens=20,
            )
        )
        cleaned_query = response.text.strip().replace('"', '').replace("'", "")
        return cleaned_query
    except Exception:
        return raw_query # Jika gagal, kembalikan kueri mentah agar diurus Sastrawi

def engine_retrieval(query, data, user_lat, user_lon, kategori_pilihan, mode):
    if not query.strip():
        if kategori_pilihan != "Semua Kategori" and 'kategori' in data.columns:
            return data[data['kategori'].astype(str).str.lower() == kategori_pilihan.lower()], "menampilkan_kategori_saja", ""
        return data, "menampilkan_kategori_saja", ""

    # Mencegat kueri mentah dengan Gemini LLM untuk koreksi typo dan dialek
    query_diproses_llm = preprocess_query_with_llm(query, mode)
    st.session_state.preprocessed_query = query_diproses_llm # Simpan untuk referensi UI

    stemmer, stopwords_final = inisialisasi_nlp()
    query_lower = query_diproses_llm.lower()
    
    # Kamus ontologi statis sebagai jaring pengaman (fallback) untuk typo umum
    kamus_ontologi = {
        'wrg': 'warung', 'wrng': 'warung', 'rm': 'warung', 'warunk': 'warung', 'warunhq': 'warung', 'warong': 'warung',
        'bngkl': 'bengkel', 'bgkl': 'bengkel', 'bengkl': 'bengkel',
        'jln': 'jl', 'jalan': 'jl', 'gg': 'jl', 'dkt': 'dekat', 'dkat': 'dekat',
        'fc': 'fotokopi', 'fotocopy': 'fotokopi', 'mart': 'minimarket',
        'supermarket': 'minimarket', 'apotik': 'apotek'
    }

    # 1. Ekstraksi Token
    words = re.findall(r'\w+', query_lower)
    
    # 2. Normalisasi Ontologi (Fallback)
    words_normalized = [kamus_ontologi.get(w, w) for w in words]
    
    # 3. Hapus Stopwords
    words_filtered = [w for w in words_normalized if w not in stopwords_final]

    # 4. STEMMING TOTAL 
    stemmed_words = [stemmer.stem(w) for w in words_filtered]

    # 5. Deteksi Niat Spasial Pasca-Stemming
    kata_spasial = {'dekat', 'sekitar', 'sini', 'area', 'wilayah'}
    niat_terdekat = any(kata in stemmed_words for kata in kata_spasial)

    # 6. Isolasi Kata Kunci Bersih
    keywords = [k for k in stemmed_words if k not in kata_spasial and k.strip()]

    def stem_teks(teks):
        return stemmer.stem(str(teks).lower())

    def filter_logic(dataset):
        if not keywords:
            return dataset.copy()
        dataset = dataset.copy()
        dataset['_teks_stem'] = (
            dataset['nama'].apply(stem_teks) + ' ' +
            dataset['alamat'].apply(stem_teks) + ' ' +
            dataset.get('kategori', pd.Series([''] * len(dataset), index=dataset.index)).apply(stem_teks)
        )
        mask = dataset['_teks_stem'].apply(lambda t: all(k in t for k in keywords))
        return dataset[mask].drop(columns=['_teks_stem']).copy()

    if kategori_pilihan != "Semua Kategori" and 'kategori' in data.columns:
        data_filtered = data[data['kategori'].astype(str).str.lower() == kategori_pilihan.lower()]
        hasil = filter_logic(data_filtered)
        status_override = False
        if hasil.empty:
            hasil = filter_logic(data)
            status_override = True if not hasil.empty else False
    else:
        hasil = filter_logic(data)
        status_override = False

    status_akhir = "sukses_override" if status_override else "sukses_spesifik"
    query_final = " ".join(keywords) if keywords else " ".join(stemmed_words)

    if niat_terdekat:
        if user_lat is None or user_lon is None:
            return pd.DataFrame(), "error_gps", query_final
        if not hasil.empty:
            hasil = hasil.copy()
            hasil['jarak_km'] = hitung_jarak_vektor(user_lat, user_lon, hasil['lat'], hasil['lon'])
            hasil = hasil.sort_values('jarak_km').head(3)
            return hasil, status_akhir + "_terdekat", query_final
        return hasil, "nihil_terdekat", query_final

    return hasil, status_akhir, query_final

def tanya_ai(query_bersih, df_konteks, status_sistem, kategori_pilihan, mode):
    query_tampil = query_bersih.title()
    jumlah_hasil = len(df_konteks)

    if mode == "Pakar":
        if status_sistem == "error_gps":
            return "Akses GPS ditolak atau tidak tersedia. Modul kalkulasi proksimitas dinonaktifkan."
        if status_sistem == "menampilkan_kategori_saja":
            return "Modul Pemetaan Spasial standby. Menunggu input parameter dari pengguna."
        if df_konteks.empty:
            return f"Pencarian nihil. Data UMKM untuk parameter '{query_tampil}' tidak ditemukan dalam basis data."
        teks_override = f"Peringatan: Filter kategori '{kategori_pilihan}' diabaikan karena konflik dengan kueri. " if "override" in status_sistem else ""
        if jumlah_hasil > 1:
            return f"{teks_override}Sistem mengidentifikasi {jumlah_hasil} lokasi UMKM relevan dengan '{query_tampil}'. Silakan tinjau tabel di bawah ini."
        b1 = df_konteks.iloc[0]
        jarak = f" (Jarak: {b1['jarak_km']:.2f} KM)" if 'jarak_km' in b1 else ""
        return f"{teks_override}Sistem menemukan kecocokan: {b1['nama']} di {b1['alamat']}{jarak}."

    kondisi = "KOSONG" if df_konteks.empty else "ADA"
    data_str = "Nihil" if df_konteks.empty else df_konteks.drop(columns=['cluster'], errors='ignore').head(10).to_dict('records')
    peringatan_override = "Beri tahu pengguna bahwa filter kategori diabaikan karena berbenturan dengan kata kunci pencarian." if "override" in status_sistem else "Tidak ada peringatan."

    prompt = f"""
    Bertindaklah sebagai Sistem WebGIS Profesional. Balas dengan bahasa Indonesia yang baku, logis, dan rapi.
    
    Kueri Terjemahan Mesin: "{query_tampil}" | Filter Kategori: "{kategori_pilihan}" | Status Data: {kondisi} | Total Hasil: {jumlah_hasil}
    Data Tersedia: {data_str}
    Kondisi Khusus: {peringatan_override}

    Instruksi:
    1. Jangan buat daftar list panjang di dalam teks jawaban.
    2. Jika Kondisi Khusus berlaku, masukkan peringatan tersebut di awal kalimat.
    3. Jika KOSONG: Nyatakan secara logis bahwa data UMKM dengan kata kunci tersebut tidak ada di basis data sistem.
    4. Jika ADA > 1: Laporkan jumlah lokasi yang ditemukan dan persilakan pengguna meninjau tabel di layar mereka.
    5. Jika ADA = 1: Laporkan lokasi tersebut secara ringkas.
    """

    if mode == "LLM":
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=256,
                )
            )
            return response.text
        except Exception:
            pass # Fallback ke mode pakar jika request gagal

    return tanya_ai(query_bersih, df_konteks, status_sistem, kategori_pilihan, "Pakar")

# ==========================================
# HALAMAN UTAMA: PETA UMKM
# ==========================================
if menu_pilihan == "📍 Peta UMKM (Utama)":

    # --- HERO HEADER ---
    st.markdown(f"""
    <div class="hero-header">
        <h1>📍 Pemetaan Spasial UMKM Malawele</h1>
        <div style="display:flex; align-items:center; gap:14px; margin-top:8px; flex-wrap:wrap;">
            <span class="hero-sub">WebGIS · Kelurahan Malawele · Kota Sorong · Papua Barat Daya</span>
            <span class="hero-status"><span class="dot-pulse"></span> SISTEM AKTIF</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- PANEL KONTROL PENCARIAN ---
    st.markdown('<div class="control-card">', unsafe_allow_html=True)
    col_q, col_k, col_loc = st.columns([1.6, 1, 1])

    with col_q:
        # Pindahkan form ke atas, lalu masukkan label ke dalam form
        with st.form(key="form_pencarian", clear_on_submit=False):
            # Label diletakkan di dalam form agar selalu satu grup
            st.markdown('<label style="font-family:\'Exo 2\',sans-serif; font-size:0.8rem; font-weight:600; color:var(--text-muted); letter-spacing:0.08em; text-transform:uppercase;">🔍  Konsol Pencarian</label>', unsafe_allow_html=True)
            
            sub_input, sub_btn = st.columns([5, 1])
            with sub_input:
                query_user = st.text_input("Konsol Pencarian", placeholder="cth: sa mau pi makan di warung dolo", label_visibility="collapsed")
            with sub_btn:
                st.form_submit_button("🔍", use_container_width=True)
    with col_k:
        list_kat = ["Semua Kategori"] + [str(k).title() for k in df['kategori'].dropna().unique()] if 'kategori' in df.columns else ["Semua Kategori"]
        kat_pilihan = st.selectbox("🏷  Filter Kategori", list_kat)
    with col_loc:
        mode_loc = st.radio("📡  Metode Lokasi", ["GPS Otomatis", "Input Manual"])
        u_lat, u_lon = None, None
        if mode_loc == "GPS Otomatis":
            st.markdown('<div style="background:transparent;">', unsafe_allow_html=True)
            loc = streamlit_geolocation()
            st.markdown('</div>', unsafe_allow_html=True)
            if loc and loc.get('latitude'):
                u_lat, u_lon = loc['latitude'], loc['longitude']
                st.success(f"🟢 `{u_lat:.5f}, {u_lon:.5f}`")
            else:
                st.caption("⚠️ Klik ikon target di atas untuk aktifkan GPS.")
        else:
            in_c = st.text_input("🌐  Koordinat", placeholder="-0.955, 131.305")
            if in_c:
                try:
                    u_lat, u_lon = map(float, [x.strip() for x in in_c.replace('"', '').split(',')])
                    st.success(f"🟢 `{u_lat}, {u_lon}`")
                except ValueError:
                    st.error("Format: Lat, Lon — cth: -0.955, 131.305")

    st.markdown('</div>', unsafe_allow_html=True)
    st.divider()

    # --- PROSES RETRIEVAL ---
    with st.spinner("Memproses kueri..."):
        f_df, s_status, q_bersih = engine_retrieval(query_user, df, u_lat, u_lon, kat_pilihan, mode_sistem)

    # --- PEMBAGIAN LAYAR (KIRI: PETA 70% | KANAN: AI & TABEL 30%) ---
    c_kiri, c_kanan = st.columns([2, 1], gap="large")

    # [KOLOM KIRI: VISUALISASI PETA]
    with c_kiri:
        st.markdown('<div class="map-wrapper" style="margin-top:0;">', unsafe_allow_html=True)
        st.markdown(f'<div class="map-title">🗺   LAYER PETA — {len(f_df)} TITIK LOKASI TERDETEKSI</div>', unsafe_allow_html=True)

        if (query_user or kat_pilihan != "Semua Kategori") and not f_df.empty:
            center = [f_df['lat'].mean(), f_df['lon'].mean()]
            zoom = 17 if len(f_df) <= 3 else 15
        elif u_lat:
            center, zoom = [u_lat, u_lon], 16
        else:
            center, zoom = [-0.9554, 131.3051], 14

        # Inisialisasi peta dasar (Dark Mode)
        m = folium.Map(
            location=center,
            zoom_start=zoom,
            tiles="CartoDB dark_matter",
            name="Peta Gelap"
        )

        # Penambahan Layer Satelit Google
        folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attr='Google',
            name='Satelit Google',
            overlay=False,
            control=True
        ).add_to(m)

        if u_lat:
            folium.Marker(
                [u_lat, u_lon],
                popup=folium.Popup("<b>📍 Titik Lokasi Anda</b>", max_width=150),
                icon=folium.Icon(color='black', icon='user', prefix='fa')
            ).add_to(m)

        for _, r in f_df.iterrows():
            dist_html = f"<p style='color:#F0A500; font-weight:bold; margin:4px 0;'>🚗 {r['jarak_km']:.2f} KM</p>" if 'jarak_km' in r else ""
            
            # Mengonversi nama UMKM dan wilayah ke format URL yang aman
            kueri_pencarian = f"{r['nama']} Malawele Sorong"
            nama_umkm_url = urllib.parse.quote(kueri_pencarian)
            
            html = f"""
            <div style="width:190px; font-family:'Segoe UI',sans-serif; background:#0B1A33;
                        border:1px solid #1A3560; border-radius:8px; overflow:hidden;">
                <div style="background:#112340; padding:8px 12px; border-bottom:1px solid #1A3560;">
                    <b style="color:#E8F0FE; font-size:13px;">{r['nama']}</b>
                </div>
                <div style="padding:8px 12px;">
                    <span style="font-size:11px; color:#7FA0C8;">{r['alamat']}</span>
                    {dist_html}
                    <a href="https://www.google.com/maps/search/?api=1&query={nama_umkm_url}" target="_blank"
                       style="background:#0057A8; color:#E8F0FE; padding:5px 10px; border-radius:5px;
                              text-decoration:none; display:block; font-size:11px;
                              margin-top:8px; text-align:center;">📍 Buka Navigasi</a>
                </div>
            </div>
            """
            color_map = {0: 'red', 1: 'blue', 2: 'green'}
            folium.Marker(
                [r['lat'], r['lon']],
                popup=folium.Popup(html, max_width=260),
                icon=folium.Icon(color=color_map.get(r.get('cluster', 0), 'gray'))
            ).add_to(m)

        # Mengaktifkan kontrol panel untuk mengganti mode peta
        folium.LayerControl(position='topright').add_to(m)

        st_folium(m, use_container_width=True, height=520, key="map", returned_objects=[])
        st.markdown('</div>', unsafe_allow_html=True)

   # [KOLOM KANAN: ASISTEN AI & TABEL]
    with c_kanan:
        if query_user:
            state_key = f"{q_bersih}_{kat_pilihan}_{s_status}"
            if st.session_state.last_query != state_key:
                with st.spinner("Merumuskan respons..."):
                    st.session_state.ai_response = tanya_ai(q_bersih, f_df, s_status, kat_pilihan, mode_sistem)
                    st.session_state.last_query = state_key

            # Menyuntikkan judul langsung ke dalam kotak ai-response-box
            st.markdown(f'''
            <div class="ai-response-box" style="padding: 15px;">
                <h4 style="margin-top: 0; margin-bottom: 15px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px;">
                    🤖 ASISTEN AI
                </h4>
                {st.session_state.ai_response}
            </div>
            ''', unsafe_allow_html=True)
            
            # Indikator Pre-processing LLM
            if mode_sistem == "LLM" and st.session_state.preprocessed_query != query_user:
                st.caption(f"✨ Diekstraksi menjadi: *{st.session_state.preprocessed_query}*")
                
        else:
            # Menyuntikkan judul langsung ke dalam kotak standby-box
            st.markdown("""
            <div class="standby-box" style="padding: 15px;">
                <h4 style="margin-top: 0; margin-bottom: 15px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px;">
                    🤖 ASISTEN AI
                </h4>
                <div style="text-align: center; margin-top: 20px;">
                    [ STANDBY ]<br><br>
                    Asisten navigasi AI siap merespons kueri Anda.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Menampilkan Tabel di bawah kotak AI (Masih dalam Kolom Kanan)
        if query_user and not f_df.empty:
            st.markdown(f"""
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.7rem;
                        color:#4A7AB5; letter-spacing:0.1em; margin-bottom:8px; margin-top:15px;">
                ▸ TABEL HASIL — {len(f_df)} DATA
            </div>
            """, unsafe_allow_html=True)
            cols_tampil = ['nama', 'alamat', 'jarak_km'] if 'jarak_km' in f_df.columns else ['nama', 'alamat']
            st.dataframe(f_df[cols_tampil], hide_index=True, use_container_width=True)

# ==========================================
# HALAMAN TENTANG
# ==========================================
elif menu_pilihan == "ℹ️ Tentang Web":
    st.markdown("""
    <div class="hero-header" style="margin-bottom:24px;">
        <h1>ℹ️ Arsitektur Sistem WebGIS UMKM</h1>
        <div class="hero-sub">Platform Analitik Spasial · Kelurahan Malawele · 2026</div>
    </div>
    """, unsafe_allow_html=True)

    ck, ck2 = st.columns([1, 1.2], gap="large")

    with ck:
        st.image("peta_malawele.jpeg", caption="Peta Digital Kelurahan Malawele", use_container_width=True)

        st.markdown("""
        <div style="margin-top:16px; background:rgba(0,200,255,0.04); border:1px solid rgba(0,200,255,0.15);
                    border-radius:10px; padding:16px;">
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.72rem;
                        color:#4A7AB5; letter-spacing:0.1em; margin-bottom:12px;">
                ▸ STACK TEKNOLOGI
            </div>
            <span class="badge-tech">Streamlit</span>
            <span class="badge-tech">Folium</span>
            <span class="badge-tech">Pandas</span>
            <span class="badge-tech">NumPy</span>
            <span class="badge-tech">Sastrawi NLP</span>
            <span class="badge-tech">K-Means</span>
            <span class="badge-tech">Haversine</span>
            <span class="badge-tech">Gemini AI</span>
        </div>
        """, unsafe_allow_html=True)

    with ck2:
        st.markdown('<div class="about-card">', unsafe_allow_html=True)
        st.markdown("""
        ### 📌 Ikhtisar Sistem
        Platform analitik spasial deterministik untuk klasifikasi dan penemuan rute UMKM terdekat
        di Kelurahan Malawele menggunakan kalkulasi geodesi presisi tinggi dan pemrosesan bahasa alami.

        ### ⚙️ Basis Teknologi

        **Geodesi:** Formula Haversine via operasi vektor NumPy untuk kalkulasi jarak geodetik
        dengan latensi sub-milidetik antar titik koordinat.

        **Kecerdasan Buatan:** Logika RAG Hibrida yang memadukan pra-pemrosesan model bahasa besar (Gemini) 
        untuk ekstraksi konteks dialek lokal serta koreksi pengetikan (typo), dilanjutkan penyaringan algoritma Sastrawi untuk pencarian basis data yang presisi.

        **Pengenalan Pola:** K-Means Clustering (k=3) untuk pengelompokan spasial otomatis
        berdasarkan distribusi koordinat geografis.

        **Validasi Cerdas:** Algoritma Auto-Override untuk resolusi konflik antara filter
        kategori dan parameter kueri pengguna.

        ### 👨‍💻 Tim Pengembang — Kelompok 14
        """, unsafe_allow_html=True)

        for nm, nim in [
            ("Farah Raihanun Badiu", "202555202118"),
            ("Kristous Vamel Ungirwalu", "202555202038"),
            ("Aldrin Petrus Safkaur", "202555202090"),
        ]:
            idx = [("Farah Raihanun Badiu", "202555202118"),
                   ("Kristous Vamel Ungirwalu", "202555202038"),
                   ("Aldrin Petrus Safkaur", "202555202090")].index((nm, nim)) + 1
            st.markdown(f"""
            <div class="member-item">
                <div class="member-num">{idx:02d}</div>
                <div>
                    <div style="font-weight:600; font-size:0.88rem; color:#E8F0FE;">{nm}</div>
                    <div style="font-family:'Share Tech Mono',monospace; font-size:0.7rem; color:#4A7AB5;">{nim}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:16px; padding:12px 16px; background:rgba(240,165,0,0.07);
                    border:1px solid rgba(240,165,0,0.2); border-radius:8px;">
            <span style="font-size:0.78rem; color:#7FA0C8;">Dosen Pengampu</span><br>
            <span style="font-weight:700; color:#FCD34D;">
                Fajar Rahardika Bahari Putra, S.Kom., M.Kom.
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
