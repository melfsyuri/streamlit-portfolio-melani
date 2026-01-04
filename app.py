import streamlit as st
import pandas as pd
import plotly.express as px

# Konfigurasi halaman
st.set_page_config(
    page_title="Portfolio Data – Melani",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Judul utama
st.title("📊 Portfolio Data dengan Streamlit")
st.markdown(
    """
Aplikasi ini adalah portfolio sederhana untuk menampilkan dan mengeksplorasi data dari project saya.
Silakan upload file **CSV** dari project/data yang ingin dianalisis.
"""
)

# SIDEBAR
st.sidebar.header("⚙️ Pengaturan")

uploaded_file = st.sidebar.file_uploader(
    "Upload file CSV data kamu", type=["csv"]
)

if uploaded_file is None:
    st.info("Silakan upload file CSV di sidebar untuk mulai.")
    st.stop()

# Load data
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

df = load_data(uploaded_file)

# Tampilkan informasi dasar
st.subheader("📄 Data Mentah")
show_raw = st.checkbox("Tampilkan tabel data", value=True)

if show_raw:
    st.dataframe(df)

st.markdown("---")

# Pilih kolom numerik & kategori otomatis
num_cols = list(df.select_dtypes(include="number").columns)
other_cols = [c for c in df.columns if c not in num_cols]

# Filter di sidebar
st.sidebar.subheader("🔍 Filter Data")

category_col = None
if other_cols:
    category_col = st.sidebar.selectbox(
        "Pilih kolom kategori (opsional):",
        options=["(tanpa filter)"] + other_cols,
        index=0,
    )

filtered_df = df.copy()
if category_col and category_col != "(tanpa filter)":
    unique_vals = sorted(filtered_df[category_col].dropna().unique())
    selected_vals = st.sidebar.multiselect(
        f"Nilai yang ditampilkan untuk {category_col}:",
        options=unique_vals,
        default=unique_vals,
    )
    filtered_df = filtered_df[filtered_df[category_col].isin(selected_vals)]

numeric_filter_col = None
if num_cols:
    numeric_filter_col = st.sidebar.selectbox(
        "Pilih kolom numerik untuk filter rentang (opsional):",
        options=["(tanpa filter)"] + num_cols,
        index=0,
    )

if numeric_filter_col and numeric_filter_col != "(tanpa filter)":
    min_val = float(filtered_df[numeric_filter_col].min())
    max_val = float(filtered_df[numeric_filter_col].max())
    min_sel, max_sel = st.sidebar.slider(
        f"Rentang nilai {numeric_filter_col}:",
        min_value=min_val,
        max_value=max_val,
        value=(min_val, max_val),
    )
    filtered_df = filtered_df[
        (filtered_df[numeric_filter_col] >= min_sel)
        & (filtered_df[numeric_filter_col] <= max_sel)
    ]

# VISUALISASI INTERAKTIF
st.subheader("📈 Visualisasi Interaktif")

if len(num_cols) >= 2:
    x_axis = st.selectbox("Pilih kolom untuk sumbu X:", num_cols, index=0)
    y_axis = st.selectbox("Pilih kolom untuk sumbu Y:", num_cols, index=1)

    color_opt = st.selectbox(
        "Warna berdasarkan kolom (opsional):",
        options=["(tanpa warna)"] + df.columns.to_list(),
        index=0,
    )

    chart_type = st.radio(
        "Pilih jenis grafik:",
        options=["Scatter Plot", "Line Chart", "Bar Chart"],
        horizontal=True,
    )

    if chart_type == "Scatter Plot":
        if color_opt == "(tanpa warna)":
            fig = px.scatter(filtered_df, x=x_axis, y=y_axis)
        else:
            fig = px.scatter(filtered_df, x=x_axis, y=y_axis, color=color_opt)
    elif chart_type == "Line Chart":
        if color_opt == "(tanpa warna)":
            fig = px.line(filtered_df, x=x_axis, y=y_axis)
        else:
            fig = px.line(filtered_df, x=x_axis, y=y_axis, color=color_opt)
    else:  # Bar Chart
        if color_opt == "(tanpa warna)":
            fig = px.bar(filtered_df, x=x_axis, y=y_axis)
        else:
            fig = px.bar(filtered_df, x=x_axis, y=y_axis, color=color_opt)

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning(
        "Butuh minimal 2 kolom numerik di data untuk membuat visualisasi. "
        "Coba gunakan dataset lain atau tambahkan kolom numerik."
    )

st.markdown("---")
st.header("📝 Catatan & Insight")
st.text_area(
    "Tulis insight utama dari visualisasi dan analisis yang kamu lihat di atas:",
    "",
    height=150,
)
