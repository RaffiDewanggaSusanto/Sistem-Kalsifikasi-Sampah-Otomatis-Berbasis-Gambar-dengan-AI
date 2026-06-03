import tempfile
from pathlib import Path

from flask import Flask, jsonify, request

from inference import load_model, predict_image


app = Flask(__name__)
model = load_model()


EDUCATION = {
    "kaca": {
        "jenis": "Sampah kaca",
        "pengelolaan": "Pisahkan dengan aman agar tidak melukai petugas, lalu kirim ke fasilitas daur ulang kaca.",
    },
    "kardus": {
        "jenis": "Sampah kardus",
        "pengelolaan": "Lipat kardus agar hemat ruang, pastikan kering, lalu kumpulkan untuk daur ulang.",
    },
    "kertas": {
        "jenis": "Sampah kertas",
        "pengelolaan": "Pastikan kertas tidak tercampur makanan atau minyak sebelum dikumpulkan untuk daur ulang.",
    },
    "logam": {
        "jenis": "Sampah logam",
        "pengelolaan": "Bilas kaleng atau logam ringan, lalu kumpulkan untuk bank sampah atau fasilitas daur ulang.",
    },
    "plastik": {
        "jenis": "Sampah plastik",
        "pengelolaan": "Bersihkan, keringkan, dan kumpulkan untuk bank sampah atau fasilitas daur ulang.",
    },
    "residu": {
        "jenis": "Sampah residu",
        "pengelolaan": "Buang ke tempat sampah residu karena umumnya sulit didaur ulang dan perlu penanganan khusus.",
    },
}


def get_education(label):
    key = label.lower()
    return EDUCATION.get(
        key,
        {
            "jenis": label,
            "pengelolaan": "Periksa kembali kategori sampah dan buang sesuai panduan lokal.",
        },
    )


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Field file harus bernama 'image'."}), 400

    uploaded_file = request.files["image"]
    suffix = Path(uploaded_file.filename or "image.jpg").suffix or ".jpg"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as temp_file:
        uploaded_file.save(temp_file.name)
        predictions = predict_image(temp_file.name, model=model, top_k=3)

    top_prediction = predictions[0]
    return jsonify(
        {
            "prediction": top_prediction,
            "top_3": predictions,
            "education": get_education(top_prediction["label"]),
            "disclaimer": "Hasil klasifikasi adalah rekomendasi AI, bukan keputusan mutlak.",
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
