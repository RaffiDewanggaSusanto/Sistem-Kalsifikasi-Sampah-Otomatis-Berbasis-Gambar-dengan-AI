import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(
    page_title="Dashboard Klasifikasi Sampah",
    page_icon="♻️",
    layout="wide"
)

st.title("♻️ Dashboard Analisis Dataset Klasifikasi Sampah 2026")

# =====================
# LOAD DATA
# =====================

df_kelas = pd.read_csv("distribusi_kelas.csv")
df_cleaning = pd.read_csv("perbandingan_cleaning.csv")
df_ratio = pd.read_csv("aspect_ratio.csv")
cm_df = pd.read_csv("confusion_matrix.csv", index_col=0)

# =====================
# SIDEBAR
# =====================

st.sidebar.title("Navigasi")

menu = st.sidebar.radio(
    "Pilih Halaman",
    [
        "🏠 Overview",
        "🎯 Meningkatkan Akurasi Klasifikasi",
        "🧹 Proses Cleaning Data",
        "⚠️ Kesalahan Klasifikasi Sampah"
    ]
)

st.sidebar.markdown("## 📂 Dataset")

st.sidebar.write("""
**Jumlah Kelas:** 6

- 📄 Kertas
- 📦 Kardus
- 🥤 Plastik
- 🍾 Kaca
- 🔩 Logam
- 🗑️ Residu
""")

st.markdown("""
<style>

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #009688,
        #004D40
    );
}

[data-testid="stSidebar"] * {
    color: white;
}

</style>
""", unsafe_allow_html=True)

# =====================
# OVERVIEW
# =====================

if menu == "🏠 Overview":

    st.header("Ringkasan Dataset")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        label="📦 Dataset Setelah Balancing",
        value="4.993"
    )

    col2.metric(
        label="🔄 Dataset Hasil Augmentasi",
        value="4.907"
    )

    col3.metric(
        label="♻️ Kategori Sampah",
        value="6"
    )

    st.subheader("Distribusi Data Tiap Kategori")

    fig, ax = plt.subplots(figsize=(8,4))

    ax.bar(
        df_kelas["Kategori"],
        df_kelas["Jumlah"]
    )

    ax.set_ylabel("Jumlah Gambar")

    st.pyplot(fig)

    st.info("""
   📌 Insight:
      - Dataset terdiri dari 6 kategori sampah dengan total 4.993 gambar setelah proses cleaning.
      - Kategori Kertas memiliki jumlah data terbanyak (1.272 gambar).
      - Kategori Kardus memiliki jumlah data paling sedikit (514 gambar).     
      - Dataset hasil augmentasi digunakan untuk menyeimbangkan jumlah data
        antar kategori sehingga beberapa kategori mengalami penambahan data,
        sementara distribusi keseluruhan menjadi lebih seimbang.
     """)
    
    st.success("""
    📋 Ringkasan Dataset

     Dataset klasifikasi sampah terdiri dari 6 kategori utama dengan
     total 4.993 gambar setelah proses cleaning. Distribusi data masih
     belum sepenuhnya seimbang karena kategori Kertas memiliki jumlah
     data paling banyak, sedangkan Kardus memiliki jumlah data paling
     sedikit.
     """)
    
# =====================
# DATA DICTIONARY
# =====================
    
    st.subheader("📖 Data Dictionary")

    st.write("""
    Data Dictionary berikut menjelaskan variabel yang digunakan
    dalam analisis dataset klasifikasi sampah dan proses evaluasi model.
    """)

    data_dict = pd.DataFrame({
        "Variabel": [
            "Kertas",
            "Kardus",
            "Plastik",
            "Kaca",
            "Logam",
            "Residu"
        ],

        "Tipe Data" : [
            "Kategori",
            "Kategori",
            "Kategori",
            "Kategori",
            "Kategori",
            "Kategori"
        ],

        "Deskripsi" : [
            "Kategori sampah berbahan dasar kertas",
            "Kategori sampah berbahan dasar kardus",
            "Kategori sampah berbahan dasar plastik",
            "Kategori sampah berbahan dasar kaca",
            "Kategori sampah berbahan dasar logam",
            "Kategori sampah yang tidak dapat didaur ulang"
        ]
    })

    st.dataframe(
        data_dict,
        use_container_width=True
    )
    
# =====================
# PERTANYAAN 1
# =====================

elif menu == "🎯 Meningkatkan Akurasi Klasifikasi":

    fig, ax = plt.subplots(figsize=(8,4))

    ax.bar(
        df_kelas["Kategori"],
        df_kelas["Jumlah"]
    )

    ax.set_title("Distribusi Dataset per Kategori")
    ax.set_ylabel("Jumlah Gambar")

    st.pyplot(fig)

    st.info("""
    📌 Insight:

     - Kategori Kertas memiliki jumlah data terbanyak yaitu 1.272 gambar.
     - Kategori Kardus memiliki jumlah data paling sedikit yaitu 514 gambar.
     - Perbedaan jumlah data antar kategori menunjukkan dataset belum sepenuhnya seimbang.
     - Ketidakseimbangan data dapat menyebabkan model lebih mudah mengenali kategori dengan jumlah data yang lebih banyak dibanding kategori dengan jumlah data yang sedikit.
    """)

    st.success("""
    🎯 Kesimpulan:

     Untuk mencapai target akurasi minimal 90%, diperlukan upaya untuk
     menyeimbangkan distribusi data antar kategori melalui augmentasi
     atau penambahan data pada kategori yang jumlahnya masih sedikit,
     seperti Kardus dan Residu. Dataset yang lebih seimbang dapat
     membantu model belajar lebih baik pada seluruh kategori sampah.
     """)


# =====================
# PERTANYAAN 2
# =====================

elif menu == "🧹 Proses Cleaning Data":

    st.dataframe(df_cleaning)

    fig, ax = plt.subplots(figsize=(6,4))

    ax.bar(
        df_cleaning["Kondisi"],
        df_cleaning["Jumlah Data"]
    )

    ax.set_ylabel("Jumlah Data")

    st.pyplot(fig)

    before = 5200
    after = 4993

    removed = before - after

    persen = round(
        removed / before * 100,
        2
    )

    st.metric(
        "Persentase Data Dihapus",
        f"{persen}%"
    )

    st.info(f"""
    📌 Insight:

     - Dataset awal berjumlah {before} gambar.
     - Setelah cleaning tersisa {after} gambar.
     - Sebanyak {removed} gambar dihapus karena rusak, duplikat,
       atau tidak sesuai standar.
     - Persentase data yang dihapus sebesar {persen}%.
     """)

    st.subheader("Distribusi Aspect Ratio")

    fig, ax = plt.subplots(figsize=(7,4))

    ax.hist(
        df_ratio["Aspect_Ratio"],
        bins=20
    )

    ax.set_xlabel("Aspect Ratio")
    ax.set_ylabel("Jumlah")

    st.pyplot(fig)

    st.success("""
    🧹 Kesimpulan:

     Proses cleaning berhasil meningkatkan kualitas dataset dengan
     menghilangkan data yang tidak valid dan memastikan gambar memiliki
     karakteristik yang lebih konsisten sebelum digunakan untuk pelatihan model.
     """)

# =====================
# PERTANYAAN 3
# =====================

elif menu == "⚠️ Kesalahan Klasifikasi Sampah":

    st.dataframe(cm_df)

    fig, ax = plt.subplots(figsize=(8,6))

    im = ax.imshow(cm_df)

    ax.set_xticks(np.arange(len(cm_df.columns)))
    ax.set_yticks(np.arange(len(cm_df.index)))

    ax.set_xticklabels(cm_df.columns)
    ax.set_yticklabels(cm_df.index)

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right"
    )

    for i in range(len(cm_df.index)):
        for j in range(len(cm_df.columns)):
            ax.text(
                j,
                i,
                cm_df.iloc[i, j],
                ha="center",
                va="center"
            )

    plt.colorbar(im)

    st.pyplot(fig)

    st.warning("""
    ⚠️ Insight:

     Berdasarkan confusion matrix hasil evaluasi sampel data,
     kesalahan klasifikasi ditemukan pada kategori Kertas dan Kardus.

     Hal ini menunjukkan bahwa kedua kategori memiliki karakteristik
     visual yang mirip sehingga lebih sulit dibedakan oleh model
     dibandingkan kategori lainnya.
     """)
    
    st.success("""
    📌 Kesimpulan:

     Kategori yang memiliki karakteristik visual serupa cenderung
     menghasilkan kesalahan klasifikasi lebih tinggi.

     Penambahan data, augmentasi, dan peningkatan kualitas fitur citra
     dapat membantu mengurangi kesalahan klasifikasi pada kategori tersebut.
     """)
