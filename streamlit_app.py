
import streamlit as st
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
from PIL import Image
import sqlite3
import numpy as np
import json
import io

st.set_page_config(
    page_title="Palm Ripeness Detection",
    page_icon="🌴",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATABASE = BASE_DIR / "history.db"

CLASS_NAMES = [
    "unripe",
    "ripe",
    "overripe",
    "empty"
]

def get_connection():
    return sqlite3.connect(DATABASE)

def init_database():
    connection = get_connection()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS detection_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            model TEXT NOT NULL,
            total INTEGER NOT NULL,
            counts TEXT NOT NULL,
            detections TEXT NOT NULL,
            created_at TEXT NOT NULL,
            image_blob BLOB
        )
    """)

    connection.commit()
    connection.close()

def save_history(
    filename,
    model_name,
    total,
    counts,
    detections,
    image_bytes
):
    connection = get_connection()

    connection.execute("""
        INSERT INTO detection_history
        (filename, model, total, counts, detections,
         created_at, image_blob)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        filename,
        model_name,
        total,
        json.dumps(counts),
        json.dumps(detections),
        datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        image_bytes
    ))

    connection.commit()
    connection.close()

def get_history():
    connection = get_connection()
    connection.row_factory = sqlite3.Row

    histories = connection.execute("""
        SELECT * FROM detection_history
        ORDER BY id DESC
    """).fetchall()

    connection.close()
    return histories

init_database()

@st.cache_resource
def load_model(model_name):
    model_path = MODEL_DIR / f"{model_name}.pt"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {model_path}"
        )

    return YOLO(str(model_path))


st.sidebar.title("🌴 Palm Ripeness")
page = st.sidebar.radio(
    "Menu",
    ["Detection", "Dashboard", "History"]
)

st.sidebar.caption(
    "Palm Fruit Ripeness Detection"
)

if page == "Detection":

    st.title("🌴 Palm Fruit Ripeness Detection")
    st.write(
        "Deteksi tingkat kematangan buah sawit "
        "menggunakan model YOLO."
    )

    model_name = st.selectbox(
        "Pilih Model",
        ["yolov8s", "yolo11s"]
    )

    uploaded_file = st.file_uploader(
        "Upload gambar buah sawit",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Gambar yang diupload",
            use_container_width=True
        )

        if st.button("Mulai Deteksi", type="primary"):

            try:
                with st.spinner(
                    "Memuat model dan melakukan deteksi..."
                ):

                    model = load_model(model_name)
                    image_array = np.array(image)
                    image_bgr = image_array[:, :, ::-1].copy()

                    results = model.predict(
                        source=image_bgr,
                        imgsz=416,
                        conf=0.25,
                        verbose=False
                    )

                    result = results[0]

                    annotated_bgr = result.plot()

                    annotated_rgb = annotated_bgr[:, :, ::-1]

                    counts = {
                        name: 0 for name in CLASS_NAMES
                    }

                    detections = []

                    for box in result.boxes:

                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])

                        if class_id < len(CLASS_NAMES):
                            class_name = CLASS_NAMES[class_id]
                        else:
                            class_name = str(class_id)

                        if class_name in counts:
                            counts[class_name] += 1

                        detections.append({
                            "class": class_name,
                            "confidence": round(
                                confidence * 100, 2
                            )
                        })

                    total = len(detections)

                    # Simpan gambar hasil ke memory
                    result_image = Image.fromarray(
                        annotated_rgb
                    )

                    image_buffer = io.BytesIO()

                    result_image.save(
                        image_buffer,
                        format="JPEG"
                    )

                    image_bytes = image_buffer.getvalue()

                    save_history(
                        filename=uploaded_file.name,
                        model_name=model_name,
                        total=total,
                        counts=counts,
                        detections=detections,
                        image_bytes=image_bytes
                    )

                st.success("Deteksi berhasil!")

                st.subheader("Hasil Deteksi")

                st.image(
                    annotated_rgb,
                    caption="Hasil deteksi YOLO",
                    use_container_width=True
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Total Objek",
                        total
                    )

                with col2:
                    st.metric(
                        "Model",
                        model_name
                    )

                st.subheader("Jumlah Berdasarkan Kelas")

                st.table(counts)

                st.subheader("Detail Deteksi")

                if detections:
                    st.dataframe(
                        detections,
                        use_container_width=True
                    )
                else:
                    st.info(
                        "Tidak ada objek yang terdeteksi."
                    )

                st.download_button(
                    "Download Hasil Deteksi",
                    data=image_bytes,
                    file_name="hasil_deteksi.jpg",
                    mime="image/jpeg"
                )

            except Exception as e:
                st.error(f"Deteksi gagal: {e}")

elif page == "Dashboard":

    st.title("📊 Dashboard")

    histories = get_history()

    total_tests = len(histories)
    total_objects = 0

    class_totals = {
        name: 0 for name in CLASS_NAMES
    }

    model_totals = {
        "yolov8s": 0,
        "yolo11s": 0
    }

    for item in histories:

        total_objects += item["total"]

        counts = json.loads(item["counts"])

        for class_name in CLASS_NAMES:
            class_totals[class_name] += counts.get(
                class_name, 0
            )

        if item["model"] in model_totals:
            model_totals[item["model"]] += 1

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Pengujian", total_tests)
    col2.metric("Total Objek", total_objects)
    col3.metric("Model Digunakan", sum(model_totals.values()))

    st.subheader("Jumlah Buah per Kelas")
    st.bar_chart(class_totals)

    st.subheader("Penggunaan Model")
    st.bar_chart(model_totals)

    st.subheader("Riwayat Terbaru")

    for item in histories[:5]:

        st.write(
            f"**{item['filename']}** — "
            f"{item['model']} — "
            f"{item['total']} objek — "
            f"{item['created_at']}"
        )

elif page == "History":

    st.title("📋 Detection History")

    histories = get_history()

    if not histories:

        st.info("Belum ada riwayat deteksi.")

    else:

        for item in histories:

            with st.expander(
                f"{item['filename']} | "
                f"{item['model']} | "
                f"{item['created_at']}"
            ):

                st.write(
                    f"Total objek: {item['total']}"
                )

                st.write(
                    "Jumlah kelas:",
                    json.loads(item["counts"])
                )

                st.write(
                    "Detail deteksi:",
                    json.loads(item["detections"])
                )

                if item["image_blob"]:

                    st.image(
                        item["image_blob"],
                        caption="Hasil deteksi",
                        use_container_width=True
                    )
