import streamlit as st

st.set_page_config(
    page_title="Volatilitas Cabai",
    page_icon="🌶️",
    layout="wide",
)

with st.sidebar:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image('cabai.png', width=60)
    st.title("Aplikasi Prediksi Volatilitas Harga Cabai")
    daftar = st.selectbox("Menu", ["Dashboard", "Prediksi Harga Cabai", "Tentang Aplikasi"])
    st.caption("dibuat dengan :fire: oleh Kelompok Barisan Terdepan")

match daftar:
    case "Dashboard":
        st.title("Dashboard Volatilitas Harga Cabai")
        st.write("Selamat datang di aplikasi prediksi volatilitas harga cabai. Aplikasi ini bertujuan untuk membantu petani dan pedagang cabai dalam `memprediksi harga cabai di masa depan`.")
        st.write("Silakan pilih menu di sidebar untuk melihat fitur-fitur yang tersedia.")
    case "Prediksi Harga Cabai":
        st.title("Prediksi Harga Cabai")
        st.write("Masukkan data harga cabai yang ingin diprediksi.")
        # Tambahkan kode untuk input data dan prediksi harga cabai
    case "Tentang Aplikasi":
        st.title("Tentang Aplikasi")
        st.write("Aplikasi ini dibuat oleh Kelompok Barisan Terdepan sebagai bagian dari tugas mata kuliah Machine Learning.")
    case _:
        st.error("Menu tidak ditemukan. Silakan pilih menu yang tersedia.")
