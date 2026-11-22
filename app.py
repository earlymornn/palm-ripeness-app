
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime
import io

# =========================
# PAGE CONFIGURATION
# =========================
st.set_page_config(
    page_title="Palm Ripeness Detection",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
/* Main app background */
.stApp {
    background-color: #f6f8f4;
    color: #26352b;
}

/* Main content spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Main headings */
h1, h2, h3 {
    color: #254d32 !important;
    font-weight: 700 !important;
}

/* Paragraphs */
p, label {
    color: #435347;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #e8efe5;
    border-right: 1px solid #d5e0d0;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #254d32 !important;
}

/* Sidebar navigation radio */
[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 8px;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    background-color: #f8faf6;
    border-radius: 10px;
    padding: 10px 14px;
    border: 1px solid #dce6d8;
}

/* Buttons */
.stButton > button {
    background-color: #3f754b;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.2rem;
    font-weight: 600;
    transition: 0.2s ease;
}

.stButton > button:hover {
    background-color: #2d5d39;
    color: white;
    border: none;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: white;
    border: 1px dashed #9eb69b;
    border-radius: 14px;
    padding: 15px;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #e0e8dc;
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 2px 8px rgba(37, 77, 50, 0.05);
}

[data-testid="stMetricLabel"] {
    color: #68786b !important;
}

[data-testid="stMetricValue"] {
    color: #254d32 !important;
    font-weight: 700;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #e0e8dc;
    border-radius: 12px;
    overflow: hidden;
}

/* Images */
[data-testid="stImage"] img {
    border-radius: 12px;
}

/* Info, success, warning messages */
[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Horizontal divider */
hr {
    border-color: #dce5d8;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    border-radius: 10px;
}

/* Footer */
.app-footer {
    text-align: center;
    color: #829083;
    font-size: 13px;
    padding-top: 30px;
}
</style>
""", unsafe_allow_html=True)


# =========================
# PATHS AND DATABASE
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DB_PATH = BASE_DIR / "history.db"

MODEL_PATHS = {
    "YOLOv8s": MODEL_DIR / "yolov8s.pt",
    "YOLO11s": MODEL_DIR / "yolo11s.pt"
}


def init_database():
    connection = sqlite3.connect(DB_PATH)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT,
            model_name TEXT,
            prediction TEXT,
            confidence REAL,
            detection_time TEXT
        )
    """)

    connection.commit()
    connection.close()


def save_history(image_name, model_name, prediction, confidence):
    connection = sqlite3.connect(DB_PATH)

    connection.execute("""
        INSERT INTO history (
            image_name,
            model_name,
            prediction,
            confidence,
            detection_time
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        image_name,
        model_name,
        prediction,
        confidence,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    connection.commit()
    connection.close()


def get_history():
    connection = sqlite3.connect(DB_PATH)

    dataframe = pd.read_sql_query("""
        SELECT
            id,
            image_name,
            model_name,
            prediction,
            confidence,
            detection_time
        FROM history
        ORDER BY id DESC
    """, connection)

    connection.close()
    return dataframe


init_database()


# =========================
# LOAD YOLO MODEL
# =========================
@st.cache_resource
def load_model(model_path):
    return YOLO(str(model_path))


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("# 🌴 Palm Ripeness")
    st.caption("Oil Palm Fruit Detection")

    st.divider()

    page = st.radio(
        "Navigation",
        ["Detection", "Dashboard", "History"],
        index=0
    )

    st.divider()

    st.markdown("### Model")
    selected_model = st.selectbox(
        "Choose model",
        list(MODEL_PATHS.keys())
    )

    st.caption(
        "Choose the YOLO model used for detection."
    )

    st.divider()

    st.markdown(
        "<div class='app-footer'>Palm Ripeness Detection</div>",
        unsafe_allow_html=True
    )


# =========================
# PAGE: DETECTION
# =========================
if page == "Detection":

    st.title("Palm Ripeness Detection")
    st.write(
        "Upload an oil palm fruit image to detect "
        "its ripeness level."
    )

    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png"]
        )

        confidence_threshold = st.slider(
            "Confidence threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.25,
            step=0.05
        )

        detect_button = st.button(
            "Detect Ripeness",
            use_container_width=True
        )

    with col2:
        st.markdown("### Detection Result")

        if uploaded_file is None:
            st.info(
                "Upload an image first to see the detection result."
            )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        with col1:
            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

        if detect_button:
            model_path = MODEL_PATHS[selected_model]

            if not model_path.exists():
                st.error(
                    f"Model file not found: {model_path}"
                )
                st.info(
                    "Make sure the model file exists inside "
                    "the models folder."
                )

            else:
                try:
                    with st.spinner("Detecting ripeness..."):
                        model = load_model(model_path)

                        results = model.predict(
                            source=image,
                            conf=confidence_threshold,
                            verbose=False
                        )

                        result = results[0]

                        # Draw bounding boxes and labels
                        plotted_image = result.plot()
                        plotted_image = plotted_image[:, :, ::-1]

                    with col2:
                        st.image(
                            plotted_image,
                            caption="Detection Image",
                            use_container_width=True
                        )

                    boxes = result.boxes

                    if boxes is not None and len(boxes) > 0:
                        detected_items = []

                        for box in boxes:
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])

                            class_name = result.names.get(
                                class_id,
                                str(class_id)
                            )

                            detected_items.append({
                                "Class": class_name,
                                "Confidence": confidence
                            })

                        detections_df = pd.DataFrame(
                            detected_items
                        )

                        st.success(
                            f"Detection completed! "
                            f"{len(detected_items)} object(s) detected."
                        )

                        st.dataframe(
                            detections_df,
                            use_container_width=True,
                            hide_index=True
                        )

                        # Save each detected class to history
                        for item in detected_items:
                            save_history(
                                uploaded_file.name,
                                selected_model,
                                item["Class"],
                                item["Confidence"]
                            )

                    else:
                        st.warning(
                            "No palm fruit detected in this image."
                        )

                        save_history(
                            uploaded_file.name,
                            selected_model,
                            "No detection",
                            0.0
                        )

                except Exception as error:
                    st.error(
                        f"An error occurred during detection: {error}"
                    )


# =========================
# PAGE: DASHBOARD
# =========================
elif page == "Dashboard":

    st.title("Dashboard")
    st.write(
        "Summary of palm ripeness detection results."
    )

    st.divider()

    history_df = get_history()

    if history_df.empty:
        st.info(
            "There is no detection history yet. "
            "Run a detection first."
        )

    else:
        total_detections = len(history_df)

        detected_df = history_df[
            history_df["prediction"] != "No detection"
        ]

        total_detected = len(detected_df)

        average_confidence = (
            detected_df["confidence"].mean()
            if not detected_df.empty
            else 0
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Records",
            total_detections
        )

        col2.metric(
            "Detected Objects",
            total_detected
        )

        col3.metric(
            "Average Confidence",
            f"{average_confidence:.2%}"
        )

        st.divider()

        st.subheader("Ripeness Distribution")

        if not detected_df.empty:
            ripeness_counts = (
                detected_df["prediction"]
                .value_counts()
                .reset_index()
            )

            ripeness_counts.columns = [
                "Ripeness",
                "Count"
            ]

            st.bar_chart(
                ripeness_counts.set_index("Ripeness")
            )

            st.subheader("Detection by Model")

            model_counts = (
                history_df["model_name"]
                .value_counts()
                .reset_index()
            )

            model_counts.columns = [
                "Model",
                "Count"
            ]

            st.bar_chart(
                model_counts.set_index("Model")
            )


# =========================
# PAGE: HISTORY
# =========================
elif page == "History":

    st.title("Detection History")
    st.write(
        "View previous palm ripeness detection results."
    )

    st.divider()

    history_df = get_history()

    if history_df.empty:
        st.info(
            "No detection history available."
        )

    else:
        col1, col2 = st.columns([3, 1])

        with col1:
            search_text = st.text_input(
                "Search image or prediction",
                placeholder="Enter image name or ripeness..."
            )

        with col2:
            selected_filter = st.selectbox(
                "Filter by model",
                ["All Models"] + list(MODEL_PATHS.keys())
            )

        filtered_df = history_df.copy()

        if search_text:
            filtered_df = filtered_df[
                filtered_df["image_name"].str.contains(
                    search_text,
                    case=False,
                    na=False
                )
                |
                filtered_df["prediction"].str.contains(
                    search_text,
                    case=False,
                    na=False
                )
            ]

        if selected_filter != "All Models":
            filtered_df = filtered_df[
                filtered_df["model_name"] == selected_filter
            ]

        # Format confidence as percentage
        if not filtered_df.empty:
            filtered_df["confidence"] = (
                filtered_df["confidence"] * 100
            ).round(2).astype(str) + "%"

        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            f"Showing {len(filtered_df)} record(s)."
        )

        csv_data = filtered_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download History as CSV",
            data=csv_data,
            file_name="palm_ripeness_history.csv",
            mime="text/csv"
        )

        st.divider()

        if st.button(
            "Delete All History",
            type="secondary"
        ):
            connection = sqlite3.connect(DB_PATH)

            connection.execute(
                "DELETE FROM history"
            )

            connection.commit()
            connection.close()

            st.success(
                "All history has been deleted."
            )

            st.rerun()


# =========================
# FOOTER
# =========================
st.markdown("""
<div class="app-footer">
    Palm Ripeness Detection System
</div>
""", unsafe_allow_html=True)
