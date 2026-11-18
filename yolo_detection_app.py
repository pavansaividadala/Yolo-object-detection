import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
import cv2
import tempfile
import os
import io
from collections import Counter, defaultdict

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="YOLO Object Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "best (2).pt"


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)


if not Path(MODEL_PATH).is_file():
    st.error(
        f"Model file not found: {MODEL_PATH}. "
        "Place best (2).pt in the same folder as app.py."
    )
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load the YOLO model: {e}")
    st.info(
        "Install the dependencies from requirements.txt and make sure "
        "this .pt file is a compatible Ultralytics YOLO checkpoint."
    )
    st.stop()


# Use the class names stored in the trained model.
class_names = model.names
if isinstance(class_names, dict):
    class_names = {int(k): str(v) for k, v in class_names.items()}
else:
    class_names = {i: str(name) for i, name in enumerate(class_names)}


# --------------------------------------------------
# APPLICATION HEADER
# --------------------------------------------------
st.title("🎯 YOLO Object Detection")
st.caption("Detect the classes learned by your trained best (2).pt model.")
st.divider()

with st.sidebar:
    st.header("⚙️ Detection Settings")
    confidence = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=1.00,
        value=0.25,
        step=0.05
    )
    iou_threshold = st.slider(
        "IoU threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05
    )
    image_size = st.select_slider(
        "Inference image size",
        options=[320, 416, 512, 640, 768, 896, 1024],
        value=640
    )

    st.markdown("---")
    st.subheader("Model classes")
    for class_id, class_name in class_names.items():
        st.write(f"{class_id}: {class_name}")


mode = st.radio(
    "Choose inspection mode",
    ["🖼️ Image Detection", "🎬 Video Detection"],
    horizontal=True
)


# ==================================================
# IMAGE DETECTION
# ==================================================
if mode == "🖼️ Image Detection":
    st.subheader("📤 Upload an image")

    uploaded_image = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="image_upload"
    )

    if uploaded_image is not None:
        try:
            original_image = Image.open(uploaded_image).convert("RGB")
            st.image(
                original_image,
                caption="Uploaded image",
                use_container_width=True
            )

            if st.button("🔍 Detect Objects", type="primary",
                         use_container_width=True):
                with st.spinner("Running object detection..."):
                    results = model.predict(
                        source=original_image,
                        conf=confidence,
                        iou=iou_threshold,
                        imgsz=image_size,
                        verbose=False
                    )

                    result = results[0]
                    annotated_array = result.plot()
                    annotated_image = Image.fromarray(
                        cv2.cvtColor(annotated_array, cv2.COLOR_BGR2RGB)
                    )

                    detections = []
                    if result.boxes is not None:
                        for box in result.boxes:
                            class_id = int(box.cls[0].item())
                            score = float(box.conf[0].item())
                            x1, y1, x2, y2 = box.xyxy[0].tolist()

                            detections.append({
                                "Class ID": class_id,
                                "Class": class_names.get(
                                    class_id, str(class_id)
                                ),
                                "Confidence": round(score, 4),
                                "X1": round(x1, 1),
                                "Y1": round(y1, 1),
                                "X2": round(x2, 1),
                                "Y2": round(y2, 1)
                            })

                    detection_df = pd.DataFrame(
                        detections,
                        columns=[
                            "Class ID", "Class", "Confidence",
                            "X1", "Y1", "X2", "Y2"
                        ]
                    )

                    st.session_state["image_detection_result"] = {
                        "annotated": annotated_image,
                        "original": original_image,
                        "detections": detection_df
                    }

        except Exception as e:
            st.error(f"Image processing failed: {e}")

    if "image_detection_result" in st.session_state:
        data = st.session_state["image_detection_result"]
        df = data["detections"]
        total = len(df)
        class_count = df["Class"].nunique() if total else 0
        avg_conf = df["Confidence"].mean() if total else 0.0

        st.divider()
        st.subheader("📊 Detection Summary")

        c1, c2, c3 = st.columns(3)
        c1.metric("Total detections", total)
        c2.metric("Classes detected", class_count)
        c3.metric("Average confidence", f"{avg_conf:.1%}")

        left, right = st.columns(2)
        with left:
            st.markdown("**Original image**")
            st.image(data["original"], use_container_width=True)
        with right:
            st.markdown("**Detected objects**")
            st.image(data["annotated"], use_container_width=True)

        st.subheader("🔎 Detection details")
        if total:
            st.dataframe(df, use_container_width=True, hide_index=True)

            distribution = (
                df["Class"].value_counts()
                .rename_axis("Class")
                .reset_index(name="Count")
            )
            st.markdown("**Class distribution**")
            st.bar_chart(distribution, x="Class", y="Count")
        else:
            st.success(
                "No objects were detected above the selected "
                "confidence threshold."
            )

        buffer = io.BytesIO()
        data["annotated"].save(buffer, format="PNG")
        st.download_button(
            "⬇️ Download annotated image",
            data=buffer.getvalue(),
            file_name="detection_result.png",
            mime="image/png",
            use_container_width=True
        )


# ==================================================
# VIDEO DETECTION
# ==================================================
else:
    st.subheader("📤 Upload a video")

    uploaded_video = st.file_uploader(
        "Choose a video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload"
    )

    if uploaded_video is not None:
        st.video(uploaded_video)

        if st.button("🎯 Detect Objects in Video", type="primary",
                     use_container_width=True):
            input_path = None
            output_path = None
            cap = None
            writer = None

            try:
                suffix = Path(uploaded_video.name).suffix or ".mp4"
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix
                ) as input_temp:
                    input_temp.write(uploaded_video.getvalue())
                    input_path = input_temp.name

                cap = cv2.VideoCapture(input_path)
                if not cap.isOpened():
                    raise RuntimeError(
                        "OpenCV could not open this video file."
                    )

                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                if fps <= 0:
                    fps = 25.0
                if width <= 0 or height <= 0:
                    raise RuntimeError("The video has invalid dimensions.")

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp4"
                ) as output_temp:
                    output_path = output_temp.name

                writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height)
                )
                if not writer.isOpened():
                    raise RuntimeError(
                        "Could not create the output video. "
                        "Check your OpenCV/codec installation."
                    )

                progress = st.progress(0)
                status = st.empty()
                frame_number = 0
                total_detections = 0
                class_counts = Counter()
                class_confidences = defaultdict(list)

                with st.spinner("Processing video frame by frame..."):
                    while True:
                        success, frame = cap.read()
                        if not success:
                            break

                        frame_number += 1
                        result = model.predict(
                            source=frame,
                            conf=confidence,
                            iou=iou_threshold,
                            imgsz=image_size,
                            verbose=False
                        )[0]

                        annotated_frame = result.plot()
                        writer.write(annotated_frame)

                        if result.boxes is not None:
                            for box in result.boxes:
                                class_id = int(box.cls[0].item())
                                class_name = class_names.get(
                                    class_id, str(class_id)
                                )
                                score = float(box.conf[0].item())

                                total_detections += 1
                                class_counts[class_name] += 1
                                class_confidences[class_name].append(score)

                        if total_frames > 0:
                            progress.progress(
                                min(frame_number / total_frames, 1.0)
                            )
                        if frame_number % 10 == 0:
                            status.write(
                                f"Processed {frame_number} / "
                                f"{total_frames if total_frames > 0 else '?'} "
                                "frames"
                            )

                cap.release()
                cap = None
                writer.release()
                writer = None
                progress.progress(1.0)
                status.success(
                    f"Finished processing {frame_number} frames."
                )

                summary_rows = []
                for class_name, count in class_counts.items():
                    scores = class_confidences[class_name]
                    summary_rows.append({
                        "Class": class_name,
                        "Detection count": count,
                        "Average confidence": round(
                            sum(scores) / len(scores), 4
                        )
                    })

                with open(output_path, "rb") as video_file:
                    video_bytes = video_file.read()

                st.session_state["video_detection_result"] = {
                    "video_bytes": video_bytes,
                    "frames": frame_number,
                    "total": total_detections,
                    "classes": len(class_counts),
                    "avg_conf": (
                        sum(
                            sum(scores)
                            for scores in class_confidences.values()
                        ) / total_detections
                        if total_detections else 0.0
                    ),
                    "details": pd.DataFrame(summary_rows)
                }

            except Exception as e:
                st.error(f"Video processing failed: {e}")

            finally:
                if cap is not None:
                    cap.release()
                if writer is not None:
                    writer.release()
                for path in (input_path, output_path):
                    if path and os.path.exists(path):
                        os.remove(path)

    if "video_detection_result" in st.session_state:
        data = st.session_state["video_detection_result"]

        st.divider()
        st.subheader("📊 Video Detection Summary")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total detections", data["total"])
        c2.metric("Frames processed", data["frames"])
        c3.metric("Classes detected", data["classes"])
        c4.metric("Average confidence", f'{data["avg_conf"]:.1%}')

        st.subheader("🎥 Annotated video")
        st.video(data["video_bytes"])
        st.download_button(
            "⬇️ Download annotated video",
            data=data["video_bytes"],
            file_name="detection_result.mp4",
            mime="video/mp4",
            use_container_width=True
        )

        st.subheader("🔎 Class-wise detection details")
        if not data["details"].empty:
            st.dataframe(
                data["details"],
                use_container_width=True,
                hide_index=True
            )
            st.bar_chart(
                data["details"],
                x="Class",
                y="Detection count"
            )
        else:
            st.success(
                "No objects were detected above the selected "
                "confidence threshold."
            )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.caption(
    "YOLO Object Detection | Powered by Ultralytics and Streamlit"
)
