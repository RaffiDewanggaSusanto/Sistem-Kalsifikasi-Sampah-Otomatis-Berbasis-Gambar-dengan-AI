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

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
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

