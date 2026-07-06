import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

from pipeline import (
    muat_dan_proses_dataset_bawaan,
    hitung_volatilitas_baru,
    buat_mapping,
    cluster_dataset_baru,
    MODEL_PATH,
    DATA_PATH,
)

st.set_page_config(
    page_title="Volatilitas Cabai Merah",
    page_icon="🌶️",
    layout="wide",
)

st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    * { font-family: 'Inter', sans-serif; }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-weight: 700 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #e2e8f0 !important;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 500;
    }

    /* Divider */
    hr { border-color: rgba(255,255,255,0.08) !important; }

    /* Info cards */
    .info-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 16px;
    }

    /* Method step */
    .method-step {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-left: 4px solid #ef4444;
        border-radius: 0 10px 10px 0;
        padding: 16px 20px;
        margin-bottom: 12px;
    }

    /* Hero header */
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ef4444, #f97316, #eab308);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        font-weight: 400;
        margin-bottom: 24px;
    }

    /* Cluster badge */
    .cluster-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        color: white;
    }
    .badge-sangat-rendah { background: #22c55e; }
    .badge-rendah { background: #3b82f6; }
    .badge-sedang { background: #eab308; color: #1e293b; }
    .badge-tinggi { background: #f97316; }
    .badge-sangat-tinggi { background: #ef4444; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_dataset():
    """Memuat dan memproses dataset bawaan (cached)."""
    return muat_dan_proses_dataset_bawaan()


@st.cache_resource(show_spinner=False)
def load_model():
    """Memuat model KMeans (cached)."""
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def compute_volatility(df_long):
    """Menghitung volatilitas GARCH dari dataset long (cached)."""
    return hitung_volatilitas_baru(df_long, minimal_observasi=10)


@st.cache_data(show_spinner=False)
def compute_clustering(df_vol, _model):
    """Menjalankan clustering pada data volatilitas (cached)."""
    mapping = buat_mapping(_model)
    X = df_vol[['Nilai_Volatilitas']].values
    df_result = df_vol.copy()
    df_result['Cluster'] = _model.predict(X)
    df_result['Kategori Volatilitas'] = df_result['Cluster'].map(mapping)
    df_result = df_result.sort_values('Nilai_Volatilitas', ascending=False).reset_index(drop=True)
    return df_result, mapping

WARNA_KATEGORI = {
    'Sangat Rendah': '#22c55e',
    'Rendah': '#3b82f6',
    'Sedang': '#eab308',
    'Tinggi': '#f97316',
    'Sangat Tinggi': '#ef4444',
}

URUTAN_KATEGORI = ['Sangat Tinggi', 'Tinggi', 'Sedang', 'Rendah', 'Sangat Rendah']

with st.sidebar:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image('cabai.png', width=60)
    st.title("Aplikasi Prediksi Volatilitas Harga Cabai")
    st.markdown("---")
    daftar = st.selectbox(
        "Pilih Menu",
        ["Dashboard", "Analisis Data", "GARCH Volatilitas",
         "Hasil Clustering", "Simulasi Dataset Baru", "Kesimpulan"],
    )
    st.markdown("---")
    st.caption("dibuat dengan :fire: oleh **Kelompok Barisan Terdepan**")


# ╔═══════════════════════════════════════════╗
# ║          HALAMAN: DASHBOARD               ║
# ╚═══════════════════════════════════════════╝
if daftar == "Dashboard":

    # Hero
    st.markdown('<p class="hero-title">🌶️ Dashboard Volatilitas Harga Cabai</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">'
        'Aplikasi analisis dan prediksi volatilitas harga cabai di Indonesia '
        'menggunakan model GARCH(1,1) dan KMeans Clustering.'
        '</p>',
        unsafe_allow_html=True,
    )

    # Load data untuk metrik
    try:
        df = load_dataset()
        n_provinsi = df['provinsi'].nunique()
        n_komoditas = df['name'].nunique()
        tanggal_min = df['Tanggal'].min().strftime('%d %b %Y')
        tanggal_max = df['Tanggal'].max().strftime('%d %b %Y')
        n_observasi = len(df)
    except Exception:
        n_provinsi = 34
        n_komoditas = 2
        tanggal_min = "29 Apr 2024"
        tanggal_max = "29 Apr 2026"
        n_observasi = "34.000+"

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🏛️ Provinsi", n_provinsi)
    col2.metric("🌶️ Komoditas", n_komoditas)
    col3.metric("📅 Mulai", tanggal_min)
    col4.metric("📅 Berakhir", tanggal_max)

    col5, col6 = st.columns(2)
    col5.metric("📊 Total Observasi", f"{n_observasi:,}" if isinstance(n_observasi, int) else n_observasi)
    col6.metric("📆 Periode", "± 2 Tahun")

    st.markdown("---")

    # Alur Metodologi
    st.subheader("🔬 Alur Metodologi Penelitian")
    st.markdown(
        "Berikut adalah tahapan analisis volatilitas harga cabai yang digunakan dalam penelitian ini:"
    )

    steps = [
        ("1️⃣", "Data Harga Harian", "Mengumpulkan data harga cabai dari pasar tradisional di 34 provinsi Indonesia selama periode April 2024 – April 2026."),
        ("2️⃣", "Perhitungan Return", "Menghitung <em>return</em> (perubahan harga harian) sebagai persentase perubahan relatif antar hari."),
        ("3️⃣", "Model GARCH(1,1)", "Memodelkan <em>conditional volatility</em> menggunakan Generalized Autoregressive Conditional Heteroskedasticity untuk menangkap kluster volatilitas pada time-series harga."),
        ("4️⃣", "Nilai Volatilitas", "Mengekstrak rata-rata <em>conditional volatility</em> per komoditas per provinsi sebagai indikator tingkat ketidakstabilan harga."),
        ("5️⃣", "KMeans Clustering (k=5)", "Mengelompokkan provinsi berdasarkan nilai volatilitas ke dalam 5 kategori menggunakan algoritma KMeans."),
        ("6️⃣", "Kategori Volatilitas", "Memberi label kategori: <b>Sangat Rendah</b>, <b>Rendah</b>, <b>Sedang</b>, <b>Tinggi</b>, dan <b>Sangat Tinggi</b> berdasarkan urutan centroid cluster."),
    ]

    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(steps):
        with cols[i % 3]:
            st.markdown(
                f'<div class="method-step">'
                f'<strong>{icon} {title}</strong><br/>'
                f'<span style="color:#94a3b8;font-size:0.9rem;">{desc}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Tentang Aplikasi
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("🎯 Tujuan Penelitian")
        st.markdown("""
        - Menganalisis **pola volatilitas** harga cabai rawit dan cabai rawit merah di seluruh provinsi Indonesia
        - Mengidentifikasi **provinsi-provinsi dengan ketidakstabilan harga** cabai yang tinggi
        - Memberikan **rekomendasi** bagi petani, pedagang, dan pemerintah dalam mitigasi risiko harga
        - Menyediakan **tools simulasi** untuk menganalisis dataset harga cabai baru
        """)
    with col_b:
        st.subheader("📚 Teknologi yang Digunakan")
        st.markdown("""
        | Komponen | Teknologi |
        |----------|-----------|
        | Bahasa | Python 3.10+ |
        | Framework UI | Streamlit |
        | Model Volatilitas | GARCH(1,1) via `arch` library |
        | Clustering | KMeans (scikit-learn) |
        | Visualisasi | Plotly Interactive Charts |
        | Data Source | Panel Harga Pasar Tradisional |
        """)


# ╔═══════════════════════════════════════════╗
# ║         HALAMAN: ANALISIS DATA            ║
# ╚═══════════════════════════════════════════╝
elif daftar == "Analisis Data":
    st.markdown('<p class="hero-title">📊 Analisis Data Historis</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">Eksplorasi data harga cabai dari pasar tradisional di seluruh Indonesia.</p>',
        unsafe_allow_html=True,
    )

    with st.spinner("Memuat dan memproses dataset..."):
        df = load_dataset()

    # ── Filter ──
    st.subheader("🔍 Filter Data")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        provinsi_list = sorted(df['provinsi'].unique())
        selected_prov = st.multiselect(
            "Pilih Provinsi",
            provinsi_list,
            default=provinsi_list[:5],
            help="Pilih satu atau lebih provinsi untuk ditampilkan",
        )
    with col_f2:
        komoditas_list = sorted(df['name'].unique())
        selected_kom = st.multiselect(
            "Pilih Komoditas",
            komoditas_list,
            default=komoditas_list,
        )

    if not selected_prov or not selected_kom:
        st.warning("⚠️ Silakan pilih minimal satu provinsi dan satu komoditas.")
        st.stop()

    df_filtered = df[
        (df['provinsi'].isin(selected_prov)) &
        (df['name'].isin(selected_kom))
    ].copy()

    st.markdown("---")

    # ── Tab Layout ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Tren Harga", "📊 Statistik Deskriptif",
        "📉 Distribusi Return", "📋 Data Mentah"
    ])

    # TAB 1: Tren Harga
    with tab1:
        st.subheader("Tren Harga Harian")

        for komoditas in selected_kom:
            df_k = df_filtered[df_filtered['name'] == komoditas]
            if df_k.empty:
                continue

            fig = px.line(
                df_k,
                x='Tanggal', y='Harga',
                color='provinsi',
                title=f"Tren Harga {komoditas} per Provinsi",
                labels={'Harga': 'Harga (Rp)', 'Tanggal': 'Tanggal', 'provinsi': 'Provinsi'},
            )
            fig.update_layout(
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter'),
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=-0.3),
                height=480,
            )
            fig.update_yaxes(tickformat=',')
            st.plotly_chart(fig, use_container_width=True)

    # TAB 2: Statistik Deskriptif
    with tab2:
        st.subheader("Statistik Deskriptif per Provinsi & Komoditas")

        stats = (
            df_filtered.groupby(['provinsi', 'name'])['Harga']
            .agg(['mean', 'std', 'min', 'max', 'count'])
            .reset_index()
        )
        stats.columns = ['Provinsi', 'Komoditas', 'Rata-rata (Rp)', 'Std Deviasi', 'Min (Rp)', 'Max (Rp)', 'Jumlah Data']
        stats['Rata-rata (Rp)'] = stats['Rata-rata (Rp)'].round(0).astype(int)
        stats['Std Deviasi'] = stats['Std Deviasi'].round(0).astype(int)
        stats['Min (Rp)'] = stats['Min (Rp)'].round(0).astype(int)
        stats['Max (Rp)'] = stats['Max (Rp)'].round(0).astype(int)

        st.dataframe(
            stats.style.format({
                'Rata-rata (Rp)': '{:,.0f}',
                'Std Deviasi': '{:,.0f}',
                'Min (Rp)': '{:,.0f}',
                'Max (Rp)': '{:,.0f}',
            }),
            use_container_width=True,
            height=400,
        )

        # Harga rata-rata bar chart
        st.subheader("Perbandingan Harga Rata-rata")
        fig_bar = px.bar(
            stats.sort_values('Rata-rata (Rp)', ascending=True),
            x='Rata-rata (Rp)', y='Provinsi',
            color='Komoditas',
            orientation='h',
            barmode='group',
            title="Harga Rata-rata per Provinsi",
            color_discrete_sequence=['#ef4444', '#f97316'],
        )
        fig_bar.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter'),
            height=max(400, len(selected_prov) * 35),
        )
        fig_bar.update_xaxes(tickformat=',')
        st.plotly_chart(fig_bar, use_container_width=True)

    # TAB 3: Distribusi Return
    with tab3:
        st.subheader("Distribusi Return Harian (%)")
        df_ret = df_filtered.dropna(subset=['Return_100'])

        for komoditas in selected_kom:
            df_rk = df_ret[df_ret['name'] == komoditas]
            if df_rk.empty:
                continue

            fig_hist = px.histogram(
                df_rk,
                x='Return_100',
                color='provinsi',
                nbins=60,
                title=f"Distribusi Return Harian — {komoditas}",
                labels={'Return_100': 'Return (%)', 'provinsi': 'Provinsi'},
                opacity=0.7,
                barmode='overlay',
            )
            fig_hist.update_layout(
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter'),
                height=420,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    # TAB 4: Data Mentah
    with tab4:
        st.subheader("Data Mentah (Long Format)")
        st.caption(f"Menampilkan {len(df_filtered):,} baris data")

        st.dataframe(
            df_filtered[['Tanggal', 'provinsi', 'name', 'Harga', 'Return_100']]
            .rename(columns={
                'provinsi': 'Provinsi',
                'name': 'Komoditas',
                'Harga': 'Harga (Rp)',
                'Return_100': 'Return (%)',
            }),
            use_container_width=True,
            height=500,
        )

        csv_data = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download Data (CSV)",
            csv_data,
            file_name="data_cabai_filtered.csv",
            mime="text/csv",
        )


# ╔═══════════════════════════════════════════╗
# ║       HALAMAN: GARCH VOLATILITAS          ║
# ╚═══════════════════════════════════════════╝
elif daftar == "GARCH Volatilitas":
    st.markdown('<p class="hero-title">📈 Model GARCH untuk Prediksi Volatilitas</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">'
        'Implementasi model GARCH(1,1) untuk mengukur conditional volatility harga cabai.'
        '</p>',
        unsafe_allow_html=True,
    )

    # Penjelasan Teori
    with st.expander("📖 Apa itu GARCH(1,1)?", expanded=False):
        st.markdown("""
        **GARCH** (*Generalized Autoregressive Conditional Heteroskedasticity*) adalah model
        time-series yang dirancang khusus untuk memodelkan **volatilitas** (ketidakstabilan)
        pada data keuangan dan ekonomi.

        **Mengapa GARCH?**
        - Harga cabai menunjukkan fenomena **volatility clustering**: periode harga stabil
          diikuti periode fluktuasi tinggi, dan sebaliknya
        - GARCH dapat menangkap pola ini dengan memodelkan varians kondisional

        **Rumus GARCH(1,1):**

        $$\\sigma_t^2 = \\omega + \\alpha \\cdot \\varepsilon_{t-1}^2 + \\beta \\cdot \\sigma_{t-1}^2$$

        Dimana:
        - $\\sigma_t^2$ = varians kondisional pada waktu $t$
        - $\\omega$ = konstanta (baseline volatility)
        - $\\alpha$ = koefisien ARCH (dampak shock sebelumnya)
        - $\\beta$ = koefisien GARCH (persistensi volatilitas)
        - $\\varepsilon_{t-1}^2$ = kuadrat residual sebelumnya
        """)

    st.markdown("---")

    # Proses GARCH
    st.subheader("⚙️ Perhitungan Volatilitas GARCH")
    st.info(
        "💡 Proses ini menghitung **conditional volatility** menggunakan GARCH(1,1) untuk setiap "
        "kombinasi komoditas dan provinsi. Hasil disimpan di cache agar tidak perlu dihitung ulang."
    )

    with st.spinner("Memuat dataset..."):
        df = load_dataset()

    # Hitung volatilitas
    with st.spinner("Menghitung volatilitas GARCH (proses ini mungkin memakan waktu beberapa menit)..."):
        df_vol = compute_volatility(df)

    if df_vol.empty:
        st.error("❌ Tidak ada data volatilitas yang berhasil dihitung.")
        st.stop()

    st.success(f"Volatilitas berhasil dihitung untuk **{len(df_vol)} provinsi**.")

    st.markdown("---")

    # Tabel Hasil
    col_t, col_c = st.columns([1, 2])

    with col_t:
        st.subheader("📋 Tabel Volatilitas per Provinsi")
        df_vol_display = df_vol.sort_values('Nilai_Volatilitas', ascending=False).reset_index(drop=True)
        df_vol_display.index = df_vol_display.index + 1
        df_vol_display.index.name = "No"

        st.dataframe(
            df_vol_display[['Provinsi', 'Nilai_Volatilitas']]
            .rename(columns={'Nilai_Volatilitas': 'Volatilitas (%)'})
            .style.format({'Volatilitas (%)': '{:.4f}'}),
            use_container_width=True,
            height=600,
        )

    with col_c:
        st.subheader("📊 Perbandingan Volatilitas Antar Provinsi")

        fig_vol = px.bar(
            df_vol_display,
            x='Nilai_Volatilitas',
            y='Provinsi',
            orientation='h',
            title="Volatilitas GARCH per Provinsi (Descending)",
            labels={'Nilai_Volatilitas': 'Nilai Volatilitas (%)', 'Provinsi': ''},
            color='Nilai_Volatilitas',
            color_continuous_scale=['#22c55e', '#eab308', '#ef4444'],
        )
        fig_vol.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter'),
            height=max(500, len(df_vol) * 22),
            showlegend=False,
            yaxis=dict(autorange='reversed'),
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    st.markdown("---")

    # Insight
    st.subheader("💡 Insight Otomatis")
    prov_max = df_vol_display.iloc[0]
    prov_min = df_vol_display.iloc[-1]

    col_i1, col_i2 = st.columns(2)
    with col_i1:
        st.error(f"""
        **🔴 Volatilitas Tertinggi**

        **{prov_max['Provinsi']}** memiliki volatilitas tertinggi sebesar **{prov_max['Nilai_Volatilitas']:.4f}%**,
        menunjukkan bahwa harga cabai di provinsi ini sangat tidak stabil dan berfluktuasi tajam.
        """)
    with col_i2:
        st.success(f"""
        **🟢 Volatilitas Terendah**

        **{prov_min['Provinsi']}** memiliki volatilitas terendah sebesar **{prov_min['Nilai_Volatilitas']:.4f}%**,
        menunjukkan bahwa harga cabai di provinsi ini relatif stabil.
        """)


# ╔═══════════════════════════════════════════╗
# ║        HALAMAN: HASIL CLUSTERING          ║
# ╚═══════════════════════════════════════════╝
elif daftar == "Hasil Clustering":
    st.markdown('<p class="hero-title">🎯 Hasil Clustering Volatilitas</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">'
        'Pengelompokan provinsi berdasarkan tingkat volatilitas harga cabai menggunakan KMeans (k=5).'
        '</p>',
        unsafe_allow_html=True,
    )

    # Load model & data
    with st.spinner("Memuat model dan menghitung volatilitas..."):
        df = load_dataset()
        model = load_model()
        df_vol = compute_volatility(df)

    if df_vol.empty:
        st.error("❌ Tidak ada data volatilitas.")
        st.stop()

    df_result, mapping = compute_clustering(df_vol, model)

    st.markdown("---")

    # Model Info
    st.subheader("🔧 Informasi Model KMeans")
    centroid = model.cluster_centers_.flatten()
    urutan = np.argsort(centroid)
    kategori_list = ['Sangat Rendah', 'Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi']

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Algoritma", "KMeans")
    col_m2.metric("Jumlah Cluster (k)", "5")
    col_m3.metric("Jumlah Provinsi", len(df_result))

    # Centroid table
    centroid_df = pd.DataFrame({
        'Cluster ID': urutan,
        'Centroid Value': centroid[urutan],
        'Kategori': kategori_list,
    })

    st.markdown("**Centroid per Cluster (urut dari rendah ke tinggi):**")
    st.dataframe(
        centroid_df.style.format({'Centroid Value': '{:.4f}'}),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    # Hasil Clustering
    tab_cl1, tab_cl2, tab_cl3 = st.tabs([
        "📋 Tabel Hasil", "📊 Visualisasi", "📈 Distribusi Cluster"
    ])

    with tab_cl1:
        st.subheader("Tabel Hasil Clustering")

        # Color the categories
        st.dataframe(
            df_result[['Provinsi', 'Nilai_Volatilitas', 'Cluster', 'Kategori Volatilitas']]
            .rename(columns={'Nilai_Volatilitas': 'Volatilitas (%)'})
            .style.format({'Volatilitas (%)': '{:.4f}'})
            .map(
                lambda v: f"color: {WARNA_KATEGORI.get(v, '#ffffff')}; font-weight: 600"
                if isinstance(v, str) and v in WARNA_KATEGORI else "",
                subset=['Kategori Volatilitas']
            ),
            use_container_width=True,
            height=500,
        )

        csv_cl = df_result.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download Hasil Clustering (CSV)",
            csv_cl,
            file_name="hasil_clustering_volatilitas.csv",
            mime="text/csv",
        )

    with tab_cl2:
        st.subheader("Scatter Plot Volatilitas per Provinsi")

        fig_scatter = px.scatter(
            df_result,
            x='Provinsi',
            y='Nilai_Volatilitas',
            color='Kategori Volatilitas',
            size='Nilai_Volatilitas',
            title="Volatilitas per Provinsi (Berdasarkan Cluster)",
            labels={'Nilai_Volatilitas': 'Nilai Volatilitas (%)', 'Provinsi': ''},
            color_discrete_map=WARNA_KATEGORI,
            category_orders={'Kategori Volatilitas': URUTAN_KATEGORI},
        )
        fig_scatter.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter'),
            height=500,
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Bar chart by category
        st.subheader("Bar Chart per Kategori Volatilitas")
        fig_bar_cl = px.bar(
            df_result.sort_values('Nilai_Volatilitas', ascending=False),
            x='Provinsi',
            y='Nilai_Volatilitas',
            color='Kategori Volatilitas',
            title="Perbandingan Volatilitas per Provinsi",
            labels={'Nilai_Volatilitas': 'Nilai Volatilitas (%)', 'Provinsi': ''},
            color_discrete_map=WARNA_KATEGORI,
            category_orders={'Kategori Volatilitas': URUTAN_KATEGORI},
        )
        fig_bar_cl.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter'),
            height=480,
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig_bar_cl, use_container_width=True)

    with tab_cl3:
        st.subheader("Distribusi Provinsi per Kategori")

        dist = df_result['Kategori Volatilitas'].value_counts().reset_index()
        dist.columns = ['Kategori', 'Jumlah']

        col_pie, col_list = st.columns([1, 1])

        with col_pie:
            fig_donut = px.pie(
                dist,
                values='Jumlah',
                names='Kategori',
                title="Proporsi Kategori Volatilitas",
                hole=0.45,
                color='Kategori',
                color_discrete_map=WARNA_KATEGORI,
                category_orders={'Kategori': URUTAN_KATEGORI},
            )
            fig_donut.update_layout(
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter'),
                height=420,
            )
            fig_donut.update_traces(
                textposition='inside',
                textinfo='value+percent',
                textfont_size=13,
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with col_list:
            st.markdown("**Daftar Provinsi per Kategori:**")
            for kat in URUTAN_KATEGORI:
                provinsi_kat = df_result[df_result['Kategori Volatilitas'] == kat]['Provinsi'].tolist()
                if provinsi_kat:
                    warna = WARNA_KATEGORI[kat]
                    st.markdown(
                        f'<span class="cluster-badge" style="background:{warna};'
                        f'{"color:#1e293b;" if kat == "Sedang" else ""}">'
                        f'{kat} ({len(provinsi_kat)})</span>',
                        unsafe_allow_html=True,
                    )
                    st.caption(", ".join(provinsi_kat))

    st.markdown("---")

    # Interpretasi
    st.subheader("📖 Interpretasi Kategori Volatilitas")

    interpretasi = {
        'Sangat Rendah': 'Harga cabai di provinsi ini **sangat stabil**. Fluktuasi harga minimal, cocok untuk perencanaan bisnis jangka panjang.',
        'Rendah': 'Harga cabai **relatif stabil** dengan fluktuasi kecil. Risiko harga rendah bagi petani dan pedagang.',
        'Sedang': 'Terdapat **fluktuasi harga moderat**. Perlu perhatian terhadap tren musiman dan faktor supply-demand.',
        'Tinggi': 'Harga cabai **cukup berfluktuasi**. Disarankan menerapkan strategi mitigasi risiko seperti kontrak forward atau diversifikasi.',
        'Sangat Tinggi': 'Harga cabai **sangat tidak stabil** dan berisiko tinggi. Diperlukan intervensi kebijakan dan stabilisasi pasokan.',
    }

    cols_int = st.columns(5)
    for i, kat in enumerate(URUTAN_KATEGORI[::-1]):
        with cols_int[i]:
            warna = WARNA_KATEGORI[kat]
            st.markdown(
                f'<div class="info-card" style="border-top:3px solid {warna};">'
                f'<strong style="color:{warna};">{kat}</strong><br/><br/>'
                f'<span style="font-size:0.85rem;color:#94a3b8;">{interpretasi[kat]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ╔═══════════════════════════════════════════╗
# ║      HALAMAN: SIMULASI DATASET BARU       ║
# ╚═══════════════════════════════════════════╝
elif daftar == "Simulasi Dataset Baru":
    st.markdown('<p class="hero-title">🔬 Simulasi Dataset Baru</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">'
        'Upload dataset harga cabai Anda sendiri untuk dianalisis menggunakan pipeline GARCH + KMeans.'
        '</p>',
        unsafe_allow_html=True,
    )

    # Panduan Format
    with st.expander("📖 Panduan Format File", expanded=False):
        st.markdown("""
        **Format file yang didukung:** `.csv`, `.xlsx`, `.xls`

        **Struktur kolom yang diharapkan:**

        | Kolom | Keterangan | Wajib? |
        |-------|-----------|--------|
        | `no` | Nomor urut | Opsional |
        | `name` / `komoditas` | Nama komoditas (mis. "Cabai Rawit") | **Wajib** |
        | `level` | Level data | Opsional |
        | `provinsi` | Nama provinsi/daerah | Opsional* |
        | Kolom tanggal | Header berformat tanggal (DD/MM/YYYY), berisi harga | **Wajib** |

        > *Jika file tidak memiliki kolom `provinsi`, Anda **wajib** mengisi nama provinsi di bawah.

        **Contoh format wide:**
        ```
        no, name, level, 01/01/2025, 02/01/2025, ..., provinsi
        1, Cabai Rawit, 1, 45000, 46000, ..., Jawa Tengah
        ```
        """)

    st.markdown("---")

    # Upload & Config
    col_up1, col_up2 = st.columns([2, 1])

    with col_up1:
        uploaded_file = st.file_uploader(
            "📁 Upload Dataset",
            type=['csv', 'xlsx', 'xls'],
            help="Upload file harga cabai dalam format CSV atau Excel",
        )

    with col_up2:
        nama_provinsi = st.text_input(
            "🏛️ Nama Provinsi (opsional)",
            placeholder="Contoh: Jawa Tengah",
            help="Isi jika file tidak punya kolom 'provinsi'",
        )
        sheet_name_input = st.text_input(
            "📄 Nama Sheet (untuk Excel)",
            value="0",
            help="Nomor sheet (0 = pertama) atau nama sheet",
        )

    # Parse sheet_name
    try:
        sheet_name = int(sheet_name_input)
    except ValueError:
        sheet_name = sheet_name_input

    if uploaded_file is not None:
        st.markdown("---")
        st.subheader(f"📄 File: `{uploaded_file.name}`")

        if st.button("🚀 Jalankan Analisis", type="primary", use_container_width=True):

            progress_bar = st.progress(0)
            status_area = st.empty()
            log_area = st.expander("📋 Log Proses", expanded=True)
            logs = []

            def progress_callback(msg):
                logs.append(msg)
                with log_area:
                    st.text(msg)

            try:
                status_area.info("🔄 Membersihkan dataset...")
                progress_bar.progress(10)

                prov_arg = nama_provinsi.strip() if nama_provinsi.strip() else None

                try:
                    df_result = cluster_dataset_baru(
                        uploaded_file,
                        nama_provinsi=prov_arg,
                        sheet_name=sheet_name,
                        progress_callback=progress_callback,
                    )
                except ValueError as ve:
                    if "provinsi" in str(ve).lower():
                        progress_bar.progress(0)
                        status_area.warning(
                            "⚠️ File ini tidak memiliki kolom **'provinsi'**. "
                            "Silakan isi **Nama Provinsi** di kolom sebelah kanan, "
                            "lalu klik **Jalankan Analisis** kembali."
                        )
                        st.stop()
                    raise

                progress_bar.progress(100)

                if df_result is None or df_result.empty:
                    status_area.error("❌ Tidak ada data volatilitas yang berhasil dihitung. Pastikan dataset memiliki cukup data.")
                    st.stop()

                status_area.success(f"Analisis selesai! Ditemukan **{len(df_result)} provinsi/daerah**.")

                st.markdown("---")

                # Tampilkan Hasil
                st.subheader("📊 Hasil Clustering Dataset Baru")

                # Metrics
                col_r1, col_r2, col_r3 = st.columns(3)
                col_r1.metric("Jumlah Daerah", len(df_result))
                col_r2.metric("Volatilitas Tertinggi", f"{df_result['Nilai_Volatilitas'].max():.4f}%")
                col_r3.metric("Volatilitas Terendah", f"{df_result['Nilai_Volatilitas'].min():.4f}%")

                # Table
                st.dataframe(
                    df_result[['Provinsi', 'Nilai_Volatilitas', 'Cluster', 'Kategori Volatilitas']]
                    .rename(columns={'Nilai_Volatilitas': 'Volatilitas (%)'})
                    .style.format({'Volatilitas (%)': '{:.4f}'})
                    .map(
                        lambda v: f"color: {WARNA_KATEGORI.get(v, '#ffffff')}; font-weight: 600"
                        if isinstance(v, str) and v in WARNA_KATEGORI else "",
                        subset=['Kategori Volatilitas']
                    ),
                    use_container_width=True,
                )

                # Visualisasi
                col_v1, col_v2 = st.columns(2)

                with col_v1:
                    fig_s = px.bar(
                        df_result.sort_values('Nilai_Volatilitas', ascending=False),
                        x='Provinsi', y='Nilai_Volatilitas',
                        color='Kategori Volatilitas',
                        title="Volatilitas per Daerah",
                        color_discrete_map=WARNA_KATEGORI,
                        category_orders={'Kategori Volatilitas': URUTAN_KATEGORI},
                    )
                    fig_s.update_layout(
                        template='plotly_dark',
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family='Inter'),
                        height=400,
                        xaxis_tickangle=-45,
                    )
                    st.plotly_chart(fig_s, use_container_width=True)

                with col_v2:
                    dist_s = df_result['Kategori Volatilitas'].value_counts().reset_index()
                    dist_s.columns = ['Kategori', 'Jumlah']
                    fig_d = px.pie(
                        dist_s, values='Jumlah', names='Kategori',
                        title="Distribusi Kategori",
                        hole=0.45,
                        color='Kategori',
                        color_discrete_map=WARNA_KATEGORI,
                    )
                    fig_d.update_layout(
                        template='plotly_dark',
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family='Inter'),
                        height=400,
                    )
                    fig_d.update_traces(textposition='inside', textinfo='value+percent')
                    st.plotly_chart(fig_d, use_container_width=True)

                # Download
                csv_sim = df_result.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Download Hasil Simulasi (CSV)",
                    csv_sim,
                    file_name="hasil_simulasi_volatilitas.csv",
                    mime="text/csv",
                )

            except Exception as e:
                progress_bar.progress(0)
                status_area.error(f"❌ Error: {e}")
                st.exception(e)
    else:
        st.info("☝️ Silakan upload file dataset terlebih dahulu untuk memulai analisis.")


# ╔═══════════════════════════════════════════╗
# ║          HALAMAN: KESIMPULAN              ║
# ╚═══════════════════════════════════════════╝
elif daftar == "Kesimpulan":
    st.markdown('<p class="hero-title">📝 Kesimpulan Penelitian</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">'
        'Ringkasan temuan, rekomendasi, dan batasan dari analisis volatilitas harga cabai.'
        '</p>',
        unsafe_allow_html=True,
    )

    # Load data untuk kesimpulan
    try:
        df = load_dataset()
        model = load_model()
        df_vol = compute_volatility(df)
        df_result, mapping = compute_clustering(df_vol, model)
        data_loaded = True
    except Exception:
        data_loaded = False

    # Ringkasan Temuan
    st.subheader("🔍 Ringkasan Temuan")

    st.markdown("""
    Penelitian ini menganalisis **volatilitas harga cabai** di pasar tradisional Indonesia
    menggunakan pendekatan **GARCH(1,1)** untuk mengukur *conditional volatility* dan
    **KMeans Clustering (k=5)** untuk mengelompokkan provinsi berdasarkan tingkat volatilitas.
    """)

    if data_loaded and not df_result.empty:
        # Statistik per cluster
        st.subheader("📊 Ringkasan per Kategori Volatilitas")

        summary_rows = []
        for kat in ['Sangat Rendah', 'Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi']:
            subset = df_result[df_result['Kategori Volatilitas'] == kat]
            if not subset.empty:
                summary_rows.append({
                    'Kategori': kat,
                    'Jumlah Provinsi': len(subset),
                    'Rata-rata Volatilitas': f"{subset['Nilai_Volatilitas'].mean():.4f}%",
                    'Contoh Provinsi': ', '.join(subset['Provinsi'].head(3).tolist()),
                })

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # Top insights
        st.markdown("---")
        col_k1, col_k2 = st.columns(2)

        with col_k1:
            st.subheader("📈 Temuan Utama")
            n_tinggi = len(df_result[df_result['Kategori Volatilitas'].isin(['Tinggi', 'Sangat Tinggi'])])
            n_rendah = len(df_result[df_result['Kategori Volatilitas'].isin(['Rendah', 'Sangat Rendah'])])

            st.markdown(f"""
            1. **{n_tinggi} provinsi** menunjukkan volatilitas harga cabai yang **tinggi hingga sangat tinggi**,
               mengindikasikan pasar yang tidak stabil dan berisiko.

            2. **{n_rendah} provinsi** memiliki volatilitas **rendah hingga sangat rendah**,
               menunjukkan stabilitas harga yang baik.

            3. Model GARCH(1,1) berhasil menangkap pola **volatility clustering** pada
               data harga cabai di Indonesia.

            4. Pengelompokan KMeans (k=5) memberikan **segmentasi yang jelas** antara
               provinsi dengan karakteristik volatilitas berbeda.
            """)

        with col_k2:
            st.subheader("🎯 Rekomendasi")
            st.markdown("""
            **Bagi Petani & Pedagang:**
            - Provinsi dengan volatilitas tinggi: pertimbangkan **kontrak harga** atau **asuransi pertanian**
            - Diversifikasi komoditas untuk mengurangi risiko
            - Perhatikan **pola musiman** dalam fluktuasi harga

            **Bagi Pemerintah:**
            - Fokuskan **stabilisasi pasokan** di provinsi dengan volatilitas sangat tinggi
            - Kembangkan **sistem informasi harga** real-time untuk transparansi pasar
            - Pertimbangkan kebijakan **buffer stock** di daerah rawan fluktuasi
            """)

    st.markdown("---")

    # Batasan Penelitian
    st.subheader("⚠️ Batasan Penelitian")
    st.markdown("""
    - **Periode data terbatas**: Analisis menggunakan data April 2024 – April 2026 (±2 tahun).
      Periode yang lebih panjang dapat memberikan gambaran volatilitas yang lebih komprehensif.
    - **Komoditas terbatas**: Hanya mencakup Cabai Rawit dan Cabai Rawit Merah.
      Jenis cabai lainnya (cabai merah besar, cabai keriting) tidak termasuk.
    - **GARCH(1,1) sederhana**: Varian model GARCH yang lebih kompleks (EGARCH, GJR-GARCH)
      mungkin dapat menangkap efek asimetris lebih baik.
    - **Data pasar tradisional**: Harga di pasar modern/supermarket mungkin menunjukkan pola berbeda.
    """)

    # Referensi
    st.subheader("📚 Referensi Metodologi")
    st.markdown("""
    - Bollerslev, T. (1986). *Generalized Autoregressive Conditional Heteroskedasticity*.
      Journal of Econometrics, 31(3), 307-327.
    - MacQueen, J. (1967). *Some Methods for Classification and Analysis of Multivariate Observations*.
      Proceedings of the 5th Berkeley Symposium on Mathematical Statistics and Probability.
    - Sheppard, K. (2023). *arch: Autoregressive Conditional Heteroskedasticity Models in Python*.
      Python Package Documentation.
    """)

    st.markdown("---")
    st.markdown(
        '<p style="text-align:center;color:#64748b;font-size:0.9rem;">'
        '🌶️ Dibuat oleh <strong>Kelompok Barisan Terdepan</strong> — '
        'Universitas AMIKOM Yogyakarta — Mata Kuliah Machine Learning 2026'
        '</p>',
        unsafe_allow_html=True,
    )


else:
    st.error("⚠️ Menu tidak ditemukan. Silakan pilih menu yang tersedia.")
