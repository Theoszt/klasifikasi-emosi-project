Berikut adalah `README.md` yang disusun berdasarkan informasi yang telah Anda berikan:

---

````markdown
# Proyek Klasifikasi Ekspresi Wajah

## 📌 Deskripsi Proyek
Aplikasi web ini memungkinkan pengguna mengunggah gambar wajah untuk mengklasifikasikan ekspresi emosi secara otomatis. Sistem memanfaatkan model machine learning yang telah dilatih sebelumnya dan teknik pemrosesan citra seperti deteksi wajah dan ekstraksi fitur sebelum melakukan klasifikasi.

## 👥 Anggota Kelompok
- Hannia Hary Putri (077)  
- Bintang Prananda Putra (131)  
- Theopan Gerard N (227)  

## 📁 Struktur File

| File               | Deskripsi                                                                                          |
|--------------------|----------------------------------------------------------------------------------------------------|
| `app.py`           | Backend utama dengan Flask. Memuat model terlatih (PCA, SVM, encoder label), menangani upload gambar, mendeteksi wajah dengan OpenCV, mengekstrak fitur (DWT), dan mengembalikan hasil klasifikasi. |
| `Home.html`        | Halaman utama yang menjelaskan konsep klasifikasi ekspresi wajah, teknologi yang digunakan, serta aplikasinya. Didukung Tailwind CSS dan JavaScript. |
| `klasifikasi.html` | Halaman upload gambar wajah, menampilkan preview gambar, info file, serta hasil klasifikasi emosi dengan confidence score. |
| `Pelajari.html`    | Halaman edukasi yang menjelaskan tahapan klasifikasi: deteksi wajah, grayscale, augmentasi, normalisasi, DWT, dan klasifikasi SVM. |
| `index.ipynb`      | Notebook pelatihan model klasifikasi. Memuat proses training, pembuatan label, PCA, dan pelatihan SVM menggunakan dataset ekspresi wajah. |
| `Dataset dan Model`| Berisi dataset ekspresi wajah, file `pca_model.pkl`, `svm_model.pkl`, dan label yang digunakan pada backend untuk prediksi. |

## 🚀 Cara Menjalankan Aplikasi

1. **Instalasi Dependencies**
   Pastikan Python 3 dan pip telah terinstal. Kemudian install dependencies:
   ```bash
   pip install -r requirements.txt
````

2. **Jalankan Server Flask**
   Jalankan file `app.py` untuk memulai server lokal:

   ```bash
   python app.py
   ```

3. **Akses Website**
   Buka browser dan kunjungi:

   ```
   http://localhost:5000
   ```

## 🌟 Fitur Utama

* Upload gambar wajah dengan antarmuka yang user-friendly.
* Deteksi wajah otomatis menggunakan Haarcascade dari OpenCV.
* Ekstraksi fitur menggunakan **Discrete Wavelet Transform (DWT)**.
* Klasifikasi emosi dengan model **Support Vector Machine (SVM)**.
* Menampilkan hasil klasifikasi dengan confidence score.
* Halaman edukatif tentang proses klasifikasi ekspresi wajah.

## 🛠 Teknologi yang Digunakan

* **Backend:** Python 3, Flask
* **Citra & ML:** OpenCV, DWT, PCA, SVM
* **Frontend:** HTML, Tailwind CSS, JavaScript

## ⚠️ Catatan

* Pastikan gambar yang diunggah mengandung wajah yang jelas untuk hasil prediksi yang akurat.
* Model telah dilatih menggunakan dataset ekspresi wajah tertentu, sehingga akurasi dapat bervariasi tergantung pada kualitas gambar.
* File `index.ipynb` hanya digunakan untuk pelatihan model, bukan bagian dari runtime aplikasi.

---

> 📂 *Proyek ini bertujuan untuk menggabungkan teknologi machine learning dan pemrosesan citra ke dalam aplikasi berbasis web yang interaktif dan edukatif.*

```
