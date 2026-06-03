<<<<<<< HEAD
# FS Waste Classifier API

API FastAPI untuk klasifikasi gambar sampah menggunakan model TFLite dari artefak tim FS.

## Kelas

- Kaca
- Kardus
- Kertas
- Logam
- Plastik
- Residu

## Struktur

```text
app/
  main.py        # endpoint FastAPI
  inference.py   # loader model dan prediksi TFLite
  schemas.py     # response schema
  settings.py    # konfigurasi env
models/
  waste_classifier.tflite
  class_names.txt
```

## Jalankan Lokal
=======
# AI Waste Classifier

**Sistem Klasifikasi Sampah Otomatis Berbasis Gambar dengan AI**.

## Fitur yang Sudah Disiapkan

- Model TensorFlow dengan Functional API.
- Transfer learning memakai EfficientNetV2B0.
- Komponen kustom: `ChannelAttention`.
- Custom loss: `FocalLoss`.
- Custom training dan validation loop menggunakan `tf.GradientTape`.
- Logging TensorBoard untuk loss dan accuracy.
- Export model ke `.keras`, `SavedModel`, dan TFLite.
- Script inference gambar.
- REST API Flask endpoint `/predict`.
- Evaluasi dengan accuracy, precision, recall, F1-score, dan confusion matrix.

## Struktur Dataset

Siapkan dataset seperti ini:

```text
dataset/
  train/
    Kaca/
    Kardus/
    Kertas/
    Logam/
    Plastik/
    Residu/
  validation/
    Kaca/
    Kardus/
    Kertas/
    Logam/
    Plastik/
    Residu/
  test/
    Kaca/
    Kardus/
    Kertas/
    Logam/
    Plastik/
    Residu/
```

Nama folder otomatis menjadi nama kelas. Kalau kelas kalian berbeda, cukup ubah nama foldernya.

## Instalasi
>>>>>>> 278da7322045d20196e9aa52399c2320137e9cc4

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
<<<<<<< HEAD
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Dokumentasi Swagger tersedia di:

```text
http://localhost:8000/docs
```

## URL API

Jika sudah deploy ke Railway, format URL-nya seperti ini:

```text
API URL   : https://web-production-287d1.up.railway.app
Predict   : POST https://web-production-287d1.up.railway.app/predict
Health    : GET  https://web-production-287d1.up.railway.app/health
Classes   : GET  https://web-production-287d1.up.railway.app/classes
```

## Endpoint

### `GET /health`

Mengecek status API dan apakah model berhasil dimuat.

Contoh response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### `GET /model`

Mengembalikan metadata model.

Contoh response:

```json
{
  "model_type": "tflite",
  "image_size": 224,
  "classes": ["Kaca", "Kardus", "Kertas", "Logam", "Plastik", "Residu"]
}
```

### `GET /classes`

Mengembalikan daftar kelas.

Contoh response:

```json
{
  "classes": ["Kaca", "Kardus", "Kertas", "Logam", "Plastik", "Residu"]
}
```

### `POST /predict`

Upload gambar dengan field form-data bernama `file`.

Contoh cURL:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@contoh_gambar.jpg"
```

Contoh Python client:

```bash
python sample_client.py contoh_gambar.jpg
```

Contoh response:

```json
{
  "label": "Plastik",
  "confidence": 0.91,
  "predictions": [
    { "label": "Plastik", "confidence": 0.91 },
    { "label": "Residu", "confidence": 0.04 }
  ]
}
```

## Jalankan dengan Docker

```bash
docker build -t fs-waste-api .
docker run --rm -p 8000:8000 fs-waste-api
```

## Konfigurasi Env

Semua env memakai prefix `FS_`.

```text
FS_MODEL_PATH=models/waste_classifier.tflite
FS_CLASS_NAMES_PATH=models/class_names.txt
FS_IMAGE_SIZE=224
FS_MAX_UPLOAD_MB=8
```

## Catatan Deployment

- Untuk Linux container, dependency memakai `tflite-runtime` agar image lebih ringan.
- Untuk Windows lokal, `requirements.txt` memakai `tensorflow` karena paket `tflite-runtime` umumnya tidak tersedia resmi untuk Windows.
- Endpoint `/predict` menerima JPEG, PNG, WEBP, dan format gambar lain yang didukung Pillow.
=======
```

## Training

```bash
python train.py
```

Output utama:

- `artifacts/model/waste_classifier.keras`
- `artifacts/model/saved_model`
- `artifacts/class_names.txt`
- `artifacts/logs/<tanggal-training>`

## TensorBoard

```bash
tensorboard --logdir artifacts/logs
```

## Evaluasi

```bash
python evaluate.py
```

## Inference Satu Gambar

```bash
python inference.py path\ke\gambar.jpg
```

## Export TFLite

```bash
python export_tflite.py
```

## Jalankan REST API

```bash
python api.py
```

Tes endpoint:

```bash
curl -X POST http://localhost:5000/predict -F "image=@path\ke\gambar.jpg"
```

>>>>>>> 278da7322045d20196e9aa52399c2320137e9cc4
