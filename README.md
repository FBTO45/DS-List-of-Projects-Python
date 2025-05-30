# 1. Data Science Challenge with Python

## 🔍 Overview

Data Science Challenge with Python adalah tantangan atau proyek mini yang bertujuan untuk menguji dan melatih keterampilan data science menggunakan bahasa Python. Challenge ini mencakup analisis data, preprocessing, eksplorasi data (EDA), pemodelan machine learning, dan pembuatan insight dari dataset tertentu.

---

## 📅 Alur Umum Proyek

### 1. Pahami Permasalahan

Contoh: "Prediksi churn pelanggan berdasarkan data transaksi dan demografi."

### 2. Load Dataset

Gunakan pandas untuk membaca data:

```python
import pandas as pd

df = pd.read_csv("data.csv")
```

### 3. Eksplorasi Data (EDA)

* Tinjau struktur data: `df.info()`
* Statistik deskriptif: `df.describe()`
* Visualisasi:

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='churn', data=df)
plt.show()
```

### 4. Preprocessing

* Tangani missing value
* Encode data kategorik
* Normalisasi atau standardisasi

### 5. Feature Engineering

* Buat fitur baru jika perlu
* Hapus fitur yang tidak relevan

### 6. Pemodelan

Pisahkan data training dan testing:

```python
from sklearn.model_selection import train_test_split

X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Gunakan model machine learning:

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)
```

### 7. Evaluasi Model

```python
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

### 8. Insight dan Visualisasi Akhir

* Interpretasi hasil model
* Analisis fitur penting (feature importance)
* Visualisasi insight menggunakan seaborn/matplotlib

---

## 📆 Tools & Library yang Digunakan

| Tujuan           | Library                               |
| ---------------- | ------------------------------------- |
| Manipulasi Data  | `pandas`, `numpy`                     |
| Visualisasi      | `matplotlib`, `seaborn`, `plotly`     |
| Preprocessing    | `sklearn.preprocessing`               |
| Machine Learning | `scikit-learn`, `xgboost`, `lightgbm` |
| Evaluasi Model   | `sklearn.metrics`                     |

---

## 📚 Contoh Tema Challenge

* Prediksi churn pelanggan
* Prediksi harga rumah
* Klasifikasi review positif/negatif
* Deteksi penipuan transaksi
* Analisis data film (IMDb, Netflix)

---

## 🎯 Tujuan Akhir Challenge

* Melatih pemahaman pipeline data science end-to-end
* Mengasah keterampilan eksplorasi data dan visualisasi
* Meningkatkan kemampuan pembuatan model dan evaluasi
* Mengembangkan kemampuan komunikasi insight dan hasil analisis
  
---



# 2. Data Engineer Challenge with Python

## 🔍 Overview

**Data Engineer Challenge with Python** adalah tantangan praktik yang dirancang untuk menguji dan memperkuat keterampilan dalam merancang, membangun, dan mengelola pipeline data menggunakan Python. Challenge ini mencakup pemrosesan data skala besar, ETL (Extract, Transform, Load), integrasi API, manajemen database, serta penggunaan tools big data dan cloud.

---

## 📅 Alur Umum Proyek

### 1. Pahami Permasalahan

Contoh: "Bangun pipeline untuk mengambil data transaksi harian dari API, bersihkan dan masukkan ke dalam database PostgreSQL."

### 2. Ekstraksi Data

* Ambil data dari berbagai sumber:

  * File CSV/JSON/Excel
  * REST API (menggunakan `requests` atau `httpx`)
  * Database (menggunakan `sqlalchemy` atau `psycopg2`)

Contoh:

```python
import requests
response = requests.get("https://api.example.com/data")
data = response.json()
```

### 3. Transformasi Data

* Gunakan `pandas`, `pyarrow`, atau `dask` untuk membersihkan dan transformasi data:

  * Drop kolom duplikat
  * Parsing tanggal
  * Normalisasi format

Contoh:

```python
import pandas as pd

df = pd.DataFrame(data)
df['created_at'] = pd.to_datetime(df['created_at'])
df = df.drop_duplicates()
```

### 4. Load ke Database

* Gunakan `SQLAlchemy`, `psycopg2`, atau `pymysql` untuk menyimpan data ke RDBMS seperti PostgreSQL atau MySQL

Contoh:

```python
from sqlalchemy import create_engine

engine = create_engine('postgresql://user:password@localhost/dbname')
df.to_sql('transaksi', engine, index=False, if_exists='replace')
```

### 5. Orkestrasi Pipeline

* Gunakan scheduler seperti `Airflow`, `Prefect`, atau `Dagster` untuk mengatur dan menjadwalkan pipeline

### 6. Monitoring dan Logging

* Tambahkan logging menggunakan `logging`
* Simpan log ke file atau monitoring tools seperti ELK Stack, Prometheus

---

## 📚 Tools & Library yang Umum Digunakan

| Kategori              | Tools                                          |
| --------------------- | ---------------------------------------------- |
| Manipulasi Data       | `pandas`, `dask`, `pyarrow`                    |
| API & Web             | `requests`, `httpx`, `beautifulsoup4`          |
| Database              | `sqlalchemy`, `psycopg2`, `pymysql`, `sqlite3` |
| Workflow & Orkestrasi | `Airflow`, `Prefect`, `Dagster`                |
| Cloud & Big Data      | `AWS S3`, `Google BigQuery`, `Spark`, `Hadoop` |
| Logging               | `logging`, `loguru`                            |

---

## 🌐 Contoh Tema Challenge

* Pipeline ETL untuk e-commerce data
* Migrasi data dari MySQL ke PostgreSQL
* Ingest data real-time dari Kafka ke database
* Automasi scraping dan load ke warehouse
* Integrasi data dari API cuaca ke dashboard

---

## 🎯 Tujuan Challenge

* Memahami arsitektur data pipeline
* Membangun proses ETL yang efisien dan scalable
* Menghubungkan berbagai sumber data ke sistem penyimpanan
* Menjadwalkan dan memantau proses data

---



# 3. Project: Simple ETL with Pandas

## 🔍 Overview

Proyek **Simple ETL (Extract, Transform, Load) with Pandas** adalah contoh pipeline data sederhana yang menggunakan Python dan pustaka Pandas untuk mengekstrak data dari sumber (misalnya file CSV), melakukan transformasi data (pembersihan, manipulasi), dan memuat hasil akhir ke dalam file lain atau database.

---

## 📅 Tujuan Proyek

* Memahami proses dasar ETL dalam Data Engineering
* Menerapkan manipulasi data dengan Pandas
* Menghasilkan dataset bersih dan siap digunakan

---

## 📚 Struktur ETL

### 1. Extract

Mengambil data dari sumber (misal: CSV, Excel, API):

```python
import pandas as pd

df = pd.read_csv("raw_data.csv")
```

### 2. Transform

Bersihkan dan ubah data agar sesuai dengan kebutuhan analisis:

```python
# Drop kolom yang tidak relevan
df = df.drop(columns=["unnecessary_column"])

# Ganti nilai kosong
df.fillna(0, inplace=True)

# Ubah format tanggal
df["date"] = pd.to_datetime(df["date"])
```

### 3. Load

Simpan hasil akhir ke file baru atau database:

```python
# Simpan ke CSV
df.to_csv("clean_data.csv", index=False)

# Atau simpan ke database (opsional)
# from sqlalchemy import create_engine
# engine = create_engine('sqlite:///clean_data.db')
# df.to_sql('table_name', engine, index=False, if_exists='replace')
```

---

## 📆 Tools & Library

| Fungsi            | Library                          |
| ----------------- | -------------------------------- |
| Manipulasi data   | `pandas`                         |
| Ekspor file       | `csv`, `Excel`, `SQLAlchemy`     |
| Waktu dan Tanggal | `datetime`, `pandas.to_datetime` |

---

## 📈 Contoh Studi Kasus

* Bersihkan data penjualan harian dari CSV
* Gabungkan beberapa file Excel bulanan menjadi satu dataset
* Normalisasi kolom kategori dan simpan ke PostgreSQL

---

## 💪 Hasil Akhir

* File `clean_data.csv` siap digunakan untuk analisis
* Dokumentasi dan reproducible script ETL

---


# 4. Project: Machine Learning with Python - Building a Recommender System

## 🔍 Overview

Proyek ini bertujuan membangun sistem rekomendasi sederhana menggunakan Python dan algoritma machine learning. Sistem rekomendasi merupakan salah satu aplikasi machine learning yang paling populer, digunakan dalam e-commerce, streaming platform, dan layanan digital lainnya.

---

## 📅 Tujuan Proyek

* Menerapkan algoritma rekomendasi berbasis data
* Memahami pendekatan content-based dan collaborative filtering
* Mengembangkan pipeline machine learning sederhana

---

## 🔧 Jenis Sistem Rekomendasi

1. **Content-Based Filtering**

   * Rekomendasi berdasarkan kesamaan fitur item (misalnya genre film, deskripsi produk)

2. **Collaborative Filtering**

   * Rekomendasi berdasarkan pola interaksi pengguna (misalnya rating, klik, pembelian)
   * Contoh: matrix factorization menggunakan `Surprise` atau `scikit-learn`

---

## 📚 Dataset Contoh

* MovieLens dataset (`ratings.csv`, `movies.csv`)
* Produk e-commerce (user\_id, product\_id, rating)

---

## 📆 Tools & Library

| Fungsi           | Library                               |
| ---------------- | ------------------------------------- |
| Manipulasi data  | `pandas`, `numpy`                     |
| Visualisasi      | `matplotlib`, `seaborn`               |
| Machine Learning | `scikit-learn`, `Surprise`, `lightfm` |
| NLP (opsional)   | `TfidfVectorizer`                     |

---

## 🎓 Alur Proyek

### 1. Load dan Eksplorasi Data

```python
import pandas as pd
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')
```

### 2. Content-Based Filtering (Contoh dengan TF-IDF)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# TF-IDF dari genre atau deskripsi
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Cosine similarity antar item
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
```

### 3. Collaborative Filtering (Contoh dengan Surprise)

```python
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)

model = SVD()
model.fit(trainset)
predictions = model.test(testset)
```

### 4. Evaluasi Model

```python
from surprise import accuracy
print("RMSE:", accuracy.rmse(predictions))
```

### 5. Simulasi Rekomendasi

```python
def get_top_n(predictions, n=5):
    from collections import defaultdict
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n
```

---

## 📊 Output yang Dihasilkan

* Rekomendasi item untuk user tertentu
* Evaluasi model dengan RMSE atau Precision\@K
* Visualisasi kesamaan item dan rating

---

## 🚀 Pengembangan Lanjutan

* Tambahkan hybrid filtering
* Integrasi dengan Streamlit atau Flask untuk membuat web rekomendasi
* Gunakan data real-time atau data dari API

---


# 5. Project: Machine Learning with Python - Building a Recommender System with Similarity Function

## 🔍 Overview

Proyek ini bertujuan membangun sistem rekomendasi sederhana menggunakan Python dan fungsi kesamaan (similarity function). Sistem rekomendasi dengan pendekatan ini memberikan saran item kepada pengguna berdasarkan kemiripan antar item atau antar pengguna.

---

## 📅 Tujuan Proyek

* Menerapkan algoritma similarity-based recommendation
* Menggunakan teknik seperti cosine similarity dan Pearson correlation
* Memahami cara kerja sistem rekomendasi berbasis kesamaan item/user

---

## ⚙️ Jenis Sistem Rekomendasi

1. **Item-Based Similarity**

   * Mencari item yang mirip dengan item yang disukai user
   * Contoh: User suka film A → rekomendasi film yang mirip dengan A

2. **User-Based Similarity**

   * Mencari user dengan preferensi serupa → rekomendasi berdasarkan apa yang disukai user lain yang mirip

---

## 📚 Dataset Contoh

* MovieLens dataset (`ratings.csv`, `movies.csv`)
* Dataset user-product rating

---

## 📆 Tools & Library

| Fungsi          | Library                 |
| --------------- | ----------------------- |
| Manipulasi data | `pandas`, `numpy`       |
| Visualisasi     | `matplotlib`, `seaborn` |
| Similarity      | `scikit-learn`, `scipy` |

---

## 🎓 Alur Proyek

### 1. Load dan Eksplorasi Data

```python
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')
```

### 2. Buat Matrix User-Item

```python
user_item_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating')
```

### 3. Hitung Similarity Antar Item

```python
# Mengisi nilai NaN dengan 0
user_item_filled = user_item_matrix.fillna(0)

# Cosine similarity antar item
item_similarity = cosine_similarity(user_item_filled.T)
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)
```

### 4. Fungsi Rekomendasi Berdasarkan Item

```python
def recommend_items(item_id, item_similarity_df, top_n=5):
    similar_scores = item_similarity_df[item_id].sort_values(ascending=False)
    similar_scores = similar_scores.drop(item_id)
    return similar_scores.head(top_n)
```

### 5. Evaluasi (Opsional)

* Hitung Precision\@K dan Recall\@K berdasarkan user actual dan prediksi

---

## 📈 Output yang Dihasilkan

* Daftar item rekomendasi berdasarkan kesamaan item
* Matriks kesamaan item (cosine atau Pearson)
* Fungsi umum `recommend_items()` yang reusable

---

## 🚀 Pengembangan Lanjutan

* Gabungkan dengan metode collaborative filtering untuk hybrid model
* Tambahkan filter genre atau kategori untuk penyaringan hasil
* Implementasi pada aplikasi web sederhana menggunakan Streamlit

---
