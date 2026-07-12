import os
import re
import joblib
import numpy as np
import pandas as pd

from arch import arch_model
from sklearn.cluster import KMeans

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "model_kmeans_volatilitas_final.pkl")
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "harga_cabai_tradisional_full.csv")


# ==========================================================
# MEMUAT DAN MEMPROSES DATASET BAWAAN
# ==========================================================
def muat_dan_proses_dataset_bawaan(path_data=DATA_PATH):
    """
    Membaca dataset bawaan (harga_cabai_tradisional_full.csv),
    membersihkannya ke long format, dan mengembalikan DataFrame siap pakai.
    Fungsi ini dipakai oleh beberapa halaman di app.py.
    """
    if not os.path.exists(path_data):
        raise FileNotFoundError(
            f"Dataset bawaan tidak ditemukan di '{path_data}'. "
            "Pastikan file CSV ikut di-deploy bersama aplikasi."
        )

    df_raw = pd.read_csv(path_data)

    # Deteksi kolom provinsi (kolom terakhir)
    kolom_wajib_candidates = ['no', 'name', 'level', 'provinsi']
    # Normalisasi nama kolom
    mapping_kolom = {}
    for c in df_raw.columns:
        c_bersih = str(c).strip().lower()
        if c_bersih in kolom_wajib_candidates:
            mapping_kolom[c] = c_bersih
        elif 'komoditas' in c_bersih:
            mapping_kolom[c] = 'name'
    df_raw = df_raw.rename(columns=mapping_kolom)

    kolom_wajib = [c for c in ['no', 'name', 'level', 'provinsi'] if c in df_raw.columns]
    kolom_tanggal = [c for c in df_raw.columns if c not in kolom_wajib]

    # Ubah ke long format
    df_long = pd.melt(
        df_raw, id_vars=kolom_wajib, value_vars=kolom_tanggal,
        var_name='Tanggal', value_name='Harga'
    )

    df_long['Harga'] = (
        df_long['Harga'].astype(str)
        .str.replace(',', '', regex=False)
        .str.strip()
        .replace({'-': np.nan, '': np.nan, 'nan': np.nan})
    )
    df_long['Harga'] = pd.to_numeric(df_long['Harga'], errors='coerce')
    df_long['Tanggal'] = pd.to_datetime(df_long['Tanggal'], dayfirst=True, errors='coerce')
    df_long = df_long.dropna(subset=['Tanggal'])
    df_long = df_long.sort_values(['provinsi', 'name', 'Tanggal'])

    # Imputasi missing values
    df_long['Harga'] = (
        df_long.groupby(['provinsi', 'name'])['Harga']
        .transform(lambda x: x.ffill().bfill())
    )

    # Hitung return harian
    df_long['Return'] = df_long.groupby(['provinsi', 'name'])['Harga'].pct_change()
    df_long['Return_100'] = df_long['Return'] * 100

    return df_long.reset_index(drop=True)


# ==========================================================
# LATIH MODEL SEKALI SAJA (dijalankan offline/di Colab,
# hasilnya = file .pkl yang ikut di-deploy bersama app Streamlit)
# ==========================================================
def latih_dan_simpan_model_final(df_hasil_volatilitas, path_model=MODEL_PATH):

    if df_hasil_volatilitas.empty:
        raise ValueError("Data volatilitas kosong.")

    X_train = df_hasil_volatilitas[['Nilai_Volatilitas']].values

    model = KMeans(n_clusters=5, random_state=42, n_init=100)
    model.fit(X_train)

    joblib.dump(model, path_model)
    return model


# ==========================================================
# MEMBUAT MAPPING KATEGORI OTOMATIS
# ==========================================================
def buat_mapping(model):
    centroid = model.cluster_centers_.flatten()
    urutan = np.argsort(centroid)
    kategori = ['Sangat Rendah', 'Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi']
    return {cluster: kategori[i] for i, cluster in enumerate(urutan)}


# ==========================================================
# MEMBACA FILE DATASET (CSV / EXCEL, PATH ATAU FILE-LIKE OBJECT)
# ==========================================================
def _deteksi_ekstensi(file_atau_path):
    """
    Streamlit's file_uploader mengembalikan objek UploadedFile
    (bukan string path), jadi ekstensi dideteksi lewat atribut .name
    kalau bukan string biasa.
    """
    if isinstance(file_atau_path, str):
        nama = file_atau_path
    else:
        nama = getattr(file_atau_path, "name", "")
    return os.path.splitext(nama)[1].lower()


def baca_file_dataset(file_atau_path, sheet_name=0):
    ekstensi = _deteksi_ekstensi(file_atau_path)

    if ekstensi in ['.xlsx', '.xls']:
        df_raw = pd.read_excel(file_atau_path, sheet_name=sheet_name)
    elif ekstensi == '.csv':
        df_raw = pd.read_csv(file_atau_path)
    else:
        raise ValueError(
            f"Format file '{ekstensi}' tidak didukung. Gunakan .csv, .xlsx, atau .xls."
        )

    return df_raw


# ==========================================================
# MEMBERSIHKAN DATASET BARU
# ==========================================================
def bersihkan_dataset(file_atau_path, nama_provinsi=None, sheet_name=0):
    """
    nama_provinsi : WAJIB diisi kalau file tidak punya kolom 'provinsi'
                    (mis. file panel harga per-daerah tanpa label provinsi).
    """

    df_raw = baca_file_dataset(file_atau_path, sheet_name=sheet_name)

    # 1. Normalisasi nama kolom non-tanggal
    mapping_kolom = {}
    for c in df_raw.columns:
        c_bersih = str(c).strip().lower()
        if 'komoditas' in c_bersih or c_bersih == 'name':
            mapping_kolom[c] = 'name'
        elif c_bersih in ['no', 'no.']:
            mapping_kolom[c] = 'no'
        elif c_bersih == 'level':
            mapping_kolom[c] = 'level'
        elif c_bersih == 'provinsi':
            mapping_kolom[c] = 'provinsi'
    df_raw = df_raw.rename(columns=mapping_kolom)

    if 'name' not in df_raw.columns:
        raise ValueError("Kolom nama komoditas tidak ditemukan di file ini.")

    # 2. Lengkapi kolom yang tidak ada di file
    if 'provinsi' not in df_raw.columns:
        if nama_provinsi is None:
            raise ValueError(
                "File ini tidak memiliki kolom 'provinsi'. "
                "Isi nama provinsi/daerah untuk data ini terlebih dahulu."
            )
        df_raw['provinsi'] = nama_provinsi
    else:
        if nama_provinsi:
            df_raw = df_raw[df_raw['provinsi'].astype(str).str.strip().str.lower() == nama_provinsi.strip().lower()]
            if df_raw.empty:
                raise ValueError(f"Provinsi '{nama_provinsi}' tidak ditemukan dalam dataset ini.")

    if 'level' not in df_raw.columns:
        df_raw['level'] = 'item'

    if 'no' not in df_raw.columns:
        df_raw['no'] = range(1, len(df_raw) + 1)

    kolom_wajib = ['no', 'name', 'level', 'provinsi']
    kolom_tanggal_asli = [c for c in df_raw.columns if c not in kolom_wajib]

    # 3. Bersihkan header tanggal dari spasi ekstra
    rename_tanggal = {c: re.sub(r'\s+', '', str(c)) for c in kolom_tanggal_asli}
    df_raw = df_raw.rename(columns=rename_tanggal)
    kolom_tanggal = list(rename_tanggal.values())

    # 4. Ubah ke long format
    df_long = pd.melt(
        df_raw, id_vars=kolom_wajib, value_vars=kolom_tanggal,
        var_name='Tanggal', value_name='Harga'
    )

    df_long['Harga'] = (
        df_long['Harga'].astype(str)
        .str.replace(',', '', regex=False)
        .str.strip()
        .replace({'-': np.nan, '': np.nan, 'nan': np.nan})
    )
    df_long['Harga'] = pd.to_numeric(df_long['Harga'], errors='coerce')

    df_long['Tanggal'] = pd.to_datetime(df_long['Tanggal'], dayfirst=True, errors='coerce')

    df_long = df_long.sort_values(['provinsi', 'name', 'Tanggal'])

    df_long['Harga'] = (
        df_long.groupby(['provinsi', 'name'])['Harga']
        .transform(lambda x: x.ffill().bfill())
    )

    df_long['Return'] = df_long.groupby(['provinsi', 'name'])['Harga'].pct_change()
    df_long = df_long.dropna(subset=['Return'])
    df_long['Return_100'] = df_long['Return'] * 100

    return df_long.reset_index(drop=True)


# ==========================================================
# HITUNG VOLATILITAS GARCH
# ==========================================================
def hitung_volatilitas_baru(df, minimal_observasi=10, progress_callback=None):
    """
    progress_callback(pesan: str) : opsional, dipanggil tiap provinsi/komoditas
                                     selesai diproses -- berguna untuk update
                                     progress bar di Streamlit.
    """
    hasil = []

    for prov in df['provinsi'].unique():
        temp = df[df['provinsi'] == prov]
        vol = []

        for komoditas in temp['name'].unique():
            returns = temp[temp['name'] == komoditas]['Return_100'].dropna()

            if len(returns) < minimal_observasi:
                if progress_callback:
                    progress_callback(f"Dilewati: {prov} - {komoditas} (data terlalu sedikit)")
                continue

            try:
                model = arch_model(returns, mean='Constant', vol='GARCH', p=1, q=1)
                res = model.fit(disp='off')
                vol.append(res.conditional_volatility.mean())
                if progress_callback:
                    progress_callback(f"Selesai: {prov} - {komoditas}")
            except Exception as e:
                if progress_callback:
                    progress_callback(f"Gagal: {prov} - {komoditas} ({e})")

        if len(vol) > 0:
            hasil.append({'Provinsi': prov, 'Nilai_Volatilitas': np.mean(vol)})

    return pd.DataFrame(hasil)


# ==========================================================
# CLUSTER DATASET BARU (fungsi utama)
# ==========================================================
def cluster_dataset_baru(
    file_atau_path,
    nama_provinsi=None,
    path_model=MODEL_PATH,
    sheet_name=0,
    progress_callback=None
):
    if not os.path.exists(path_model):
        raise FileNotFoundError(
            "Model belum ditemukan. Pastikan file "
            f"'{os.path.basename(path_model)}' ikut di-deploy bersama aplikasi."
        )

    df_bersih = bersihkan_dataset(file_atau_path, nama_provinsi=nama_provinsi, sheet_name=sheet_name)
    df_vol = hitung_volatilitas_baru(df_bersih, progress_callback=progress_callback)

    if df_vol.empty:
        return None

    model = joblib.load(path_model)
    mapping = buat_mapping(model)

    X = df_vol[['Nilai_Volatilitas']].values
    df_vol['Cluster'] = model.predict(X)
    df_vol['Kategori Volatilitas'] = df_vol['Cluster'].map(mapping)

    df_vol = df_vol.sort_values('Nilai_Volatilitas', ascending=False).reset_index(drop=True)

    return df_vol
