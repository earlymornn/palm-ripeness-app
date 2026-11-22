
import streamlit as st
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
from PIL import Image
import sqlite3
import json
import uuid
import cv2

st.set_page_config(
    page_title="Palm Ripeness Detection",
    page_icon="🌴",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "public" / "uploads"
DATABASE = BASE_DIR / "history.db"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "unripe",
    "ripe",
    "overripe",
    "empty"
]


# Load model
@st.cache_resource
def load_model(model_path):
    return YOLO(str(model_path))


# Database
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
        (filename, model, image_path, total, counts,
         detections, created_at)
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


# Sidebar
st.sidebar.title("🌴 Palm Ripeness")
page = st.sidebar.radio(
    "Menu",
    ["Detection", "History", "Dashboard"]
)


# Detection page
if page == "Detection":

    st.title("🌴 Palm Ripeness Detection")
    st.write(
        "Deteksi tingkat kematangan buah sawit "
        "menggunakan YOLO."
    )

    model_name = st.selectbox(
        "Pilih Model",
        ["yolov8s", "yolo11s"]
    )

    model_path = MODEL_DIR / f"{model_name}.pt"

    uploaded_file = st.file_uploader(
        "Upload gambar sawit",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Gambar yang diupload",
            use_container_width=True
        )

        if st.button("🔍 Detect", type="primary"):

            if not model_path.exists():

                st.error(
                    f"Model tidak ditemukan: {model_path}"
                )

            else:

                with st.spinner("Sedang mendeteksi..."):

                    model = load_model(model_path)

                    results = model.predict(
                        source=image,
                        imgsz=416,
                        conf=0.25,
                        verbose=False
                    )

                    result = results[0]

                    annotated = result.plot()

                    annotated = cv2.cvtColor(
                        annotated,
                        cv2.COLOR_BGR2RGB
                    )

                    counts = {
                        name: 0 for name in CLASS_NAMES
                    }

                    detections = []

                    for box in result.boxes:

                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])

                        class_name = model.names[class_id]

                        if class_name not in counts:
                            counts[class_name] = 0

                        counts[class_name] += 1

                        detections.append({
                            "class": class_name,
                            "confidence": round(
                                confidence * 100, 2
                            )
                        })

                    total = len(detections)

                    output_filename = (
                        f"result_{uuid.uuid4().hex}.jpg"
                    )

                    output_path = (
                        UPLOAD_DIR / output_filename
                    )

                    cv2.imwrite(
                        str(output_path),
                        cv2.cvtColor(
                            annotated,
                            cv2.COLOR_RGB2BGR
                        )
                    )

                    save_history(
                        filename=uploaded_file.name,
                        model=model_name,
                        image_path=str(output_path),
                        total=total,
                        counts=counts,
                        detections=detections
                    )

                st.success("Deteksi berhasil!")

                st.subheader("Hasil Deteksi")

                st.image(
                    annotated,
                    use_container_width=True
                )

                st.metric(
                    "Total objek terdeteksi",
                    total
                )

                st.subheader("Jumlah setiap kelas")

                cols = st.columns(len(counts))

                for i, (name, value) in enumerate(
                    counts.items()
                ):

                    cols[i].metric(name, value)

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


# History page
elif page == "History":

    st.title("📋 Detection History")

    connection = sqlite3.connect(DATABASE)

    histories = connection.execute("""
        SELECT filename, model, total, counts, created_at
        FROM detection_history
        ORDER BY id DESC
    """).fetchall()

    connection.close()

    if histories:

        for item in histories:

            filename, model, total, counts, date = item

            with st.expander(filename):

                st.write("Model:", model)
                st.write("Tanggal:", date)
                st.write("Total objek:", total)

                st.json(json.loads(counts))

    else:

        st.info("Belum ada riwayat deteksi.")


# Dashboard page
elif page == "Dashboard":

    st.title("📊 Dashboard")

    connection = sqlite3.connect(DATABASE)

    histories = connection.execute("""
        SELECT total, counts, model
        FROM detection_history
    """).fetchall()

    connection.close()

    total_tests = len(histories)

    total_objects = sum(
        item[0] for item in histories
    )

    st.metric(
        "Total Pengujian",
        total_tests
    )

    st.metric(
        "Total Objek Terdeteksi",
        total_objects
    )

    class_totals = {
        name: 0 for name in CLASS_NAMES
    }

    for item in histories:

        counts = json.loads(item[1])

        for name, value in counts.items():

            if name not in class_totals:
                class_totals[name] = 0

            class_totals[name] += value

    st.subheader("Statistik Kelas")

    st.bar_chart(class_totals)
