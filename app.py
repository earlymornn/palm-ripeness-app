
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import sqlite3
import cv2
import uuid
import json

app = Flask(
    __name__,
    static_folder="public",
    static_url_path=""
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "public" / "uploads"
DATABASE = BASE_DIR / "history.db"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

CLASS_NAMES = ["unripe", "ripe", "overripe", "empty"]

# Model tidak langsung dimuat saat Flask dimulai
models = {}


# =========================
# LOAD MODEL SAAT DIBUTUHKAN
# =========================

def get_model(model_name):

    if model_name not in ["yolov8s", "yolo11s"]:
        raise ValueError("Model tidak valid.")

    if model_name not in models:

        model_path = MODEL_DIR / f"{model_name}.pt"

        if not model_path.exists():
            raise FileNotFoundError(
                f"File model tidak ditemukan: {model_path}"
            )

        models[model_name] = YOLO(str(model_path))

    return models[model_name]


# =========================
# DATABASE
# =========================

def init_database():

    connection = sqlite3.connect(DATABASE)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS detection_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            model TEXT NOT NULL,
            image_path TEXT NOT NULL,
            total INTEGER NOT NULL,
            counts TEXT NOT NULL,
            detections TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


def save_history(
    filename,
    model,
    image_path,
    total,
    counts,
    detections
):

    connection = sqlite3.connect(DATABASE)

    connection.execute("""
        INSERT INTO detection_history
        (
            filename,
            model,
            image_path,
            total,
            counts,
            detections,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        filename,
        model,
        image_path,
        total,
        json.dumps(counts),
        json.dumps(detections),
        datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    ))

    connection.commit()
    connection.close()


init_database()


# =========================
# VALIDASI FILE
# =========================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# =========================
# HALAMAN UTAMA
# =========================

@app.route("/")
def index():

    return render_template("index.html")


# =========================
# DETEKSI
# =========================

@app.route("/detect", methods=["POST"])
def detect():

    if "image" not in request.files:
        return jsonify({
            "error": "Gambar belum dipilih."
        }), 400

    file = request.files["image"]

    model_name = request.form.get(
        "model",
        "yolov8s"
    )

    if file.filename == "":
        return jsonify({
            "error": "Silakan pilih gambar."
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": "Format gambar harus JPG, JPEG, atau PNG."
        }), 400

    if model_name not in ["yolov8s", "yolo11s"]:
        return jsonify({
            "error": "Model tidak valid."
        }), 400

    extension = file.filename.rsplit(".", 1)[1].lower()

    filename = f"{uuid.uuid4().hex}.{extension}"

    image_path = UPLOAD_DIR / filename

    file.save(str(image_path))

    try:

        # Model baru dimuat ketika dipilih
        model = get_model(model_name)

        results = model.predict(
            source=str(image_path),
            imgsz=416,
            conf=0.25,
            verbose=False,
            device="cpu"
        )

        result = results[0]

        annotated_image = result.plot()

        output_filename = (
            f"result_{filename.rsplit('.', 1)[0]}.jpg"
        )

        output_path = UPLOAD_DIR / output_filename

        cv2.imwrite(
            str(output_path),
            annotated_image
        )

        counts = {
            name: 0 for name in CLASS_NAMES
        }

        detections = []

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = CLASS_NAMES[class_id]

            counts[class_name] += 1

            detections.append({
                "class": class_name,
                "confidence": round(
                    confidence * 100,
                    2
                )
            })

        result_image = f"/uploads/{output_filename}"

        save_history(
            filename=file.filename,
            model=model_name,
            image_path=result_image,
            total=len(detections),
            counts=counts,
            detections=detections
        )

        return jsonify({
            "success": True,
            "model": model_name,
            "image": result_image,
            "counts": counts,
            "total": len(detections),
            "detections": detections
        })

    except Exception as e:

        app.logger.exception("Deteksi gagal")

        return jsonify({
            "error": f"Deteksi gagal: {str(e)}"
        }), 500


# =========================
# HISTORY
# =========================

@app.route("/history")
def history():

    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row

    histories = connection.execute("""
        SELECT * FROM detection_history
        ORDER BY id DESC
    """).fetchall()

    connection.close()

    history_data = []

    for item in histories:

        history_data.append({
            "id": item["id"],
            "filename": item["filename"],
            "model": item["model"],
            "image_path": item["image_path"],
            "total": item["total"],
            "counts": json.loads(item["counts"]),
            "detections": json.loads(item["detections"]),
            "created_at": item["created_at"]
        })

    return render_template(
        "history.html",
        histories=history_data
    )


# =========================
# DASHBOARD
# =========================

@app.route("/dashboard")
def dashboard():

    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row

    histories = connection.execute("""
        SELECT * FROM detection_history
        ORDER BY id DESC
    """).fetchall()

    connection.close()

    total_tests = len(histories)

    total_objects = 0

    class_totals = {
        "unripe": 0,
        "ripe": 0,
        "overripe": 0,
        "empty": 0
    }

    model_totals = {
        "yolov8s": 0,
        "yolo11s": 0
    }

    for item in histories:

        total_objects += item["total"]

        counts = json.loads(item["counts"])

        for class_name in class_totals:

            class_totals[class_name] += counts.get(
                class_name,
                0
            )

        if item["model"] in model_totals:

            model_totals[item["model"]] += 1

    recent_histories = []

    for item in histories[:5]:

        recent_histories.append({
            "filename": item["filename"],
            "model": item["model"],
            "total": item["total"],
            "created_at": item["created_at"],
            "image_path": item["image_path"]
        })

    return render_template(
        "dashboard.html",
        total_tests=total_tests,
        total_objects=total_objects,
        class_totals=class_totals,
        model_totals=model_totals,
        recent_histories=recent_histories
    )


# =========================
# RUN FLASK
# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
