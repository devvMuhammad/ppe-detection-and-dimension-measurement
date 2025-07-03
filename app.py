import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
import supervision as sv
import os
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Main App Configuration ---
st.set_page_config(page_title="Computer Vision Suite", layout="wide")
st.title("Computer Vision Suite")


# --- Model and Client Caching ---
@st.cache_resource
def load_yolo_model():
    """Loads and caches the YOLOv8 segmentation model."""
    return YOLO('yolov8n-seg.pt')


# --- Session State Initialization ---
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'is_running' not in st.session_state:
    st.session_state.is_running = False

# --- Core Functions ---

def measure_dimensions(image_bytes, known_dimension_cm, ref_object_name):
    """
    Processes an image to measure object dimensions based on a reference object.
    Takes image bytes and returns an annotated image.
    """
    model = load_yolo_model()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = model(img_rgb)
    pixels_per_cm = None
    log_messages = []

    if results and results[0].masks is not None:
        class_names = results[0].names
        try:
            ref_class_id = list(class_names.keys())[list(class_names.values()).index(ref_object_name.lower())]
        except ValueError:
            log_messages.append(f"Error: Class '{ref_object_name}' not found in model.")
            log_messages.append(f"Available classes: {list(class_names.values())}")
            return None, log_messages

        reference_mask_found = False
        if results[0].boxes:
            for i, box in enumerate(results[0].boxes):
                if box.cls == ref_class_id:
                    log_messages.append(f"Found reference object: '{ref_object_name}'")
                    reference_mask = results[0].masks.xy[i]
                    reference_mask_found = True
                    break

        if not reference_mask_found:
            log_messages.append(f"Error: Reference object '{ref_object_name}' not detected.")
            return img_rgb, log_messages

        reference_contour = np.array(reference_mask, dtype=np.int32)
        ref_rect = cv2.minAreaRect(reference_contour)
        ref_pixel_dimension = max(ref_rect[1])

        if ref_pixel_dimension > 0:
            pixels_per_cm = ref_pixel_dimension / known_dimension_cm
            log_messages.append(f"Reference pixel dimension: {ref_pixel_dimension:.2f} px")
            log_messages.append(f"Known real-world dimension: {known_dimension_cm} cm")
            log_messages.append(f"Calculated pixels-per-cm: {pixels_per_cm:.2f}")
        else:
            log_messages.append("Error: Could not determine reference object's pixel dimension.")
            pixels_per_cm = None

        if pixels_per_cm:
            for i, mask in enumerate(results[0].masks.xy):
                contour = np.array(mask, dtype=np.int32)
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = box.astype(int)

                current_class_id = int(results[0].boxes[i].cls.item())
                color = (255, 0, 0) if current_class_id == ref_class_id else (0, 255, 0)
                cv2.drawContours(img_rgb, [box], 0, color, 3)

                width_px, height_px = rect[1]
                width_cm = width_px / pixels_per_cm
                height_cm = height_px / pixels_per_cm
                class_name = class_names[current_class_id]
                dim_text = f"{class_name}: {max(width_cm, height_cm):.1f}x{min(width_cm, height_cm):.1f} cm"

                font_scale = 1
                font_thickness = 2
                text_color = (255, 255, 255)
                (text_w, text_h), baseline = cv2.getTextSize(dim_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                center_x, center_y = int(rect[0][0]), int(rect[0][1])
                text_x = center_x - text_w // 2
                text_y = center_y + text_h // 2
                bg_top_left = (text_x - 5, center_y - text_h - baseline)
                bg_bottom_right = (text_x + text_w + 5, center_y)
                cv2.rectangle(img_rgb, bg_top_left, bg_bottom_right, color, cv2.FILLED)
                cv2.putText(img_rgb, dim_text, (text_x, center_y - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness, cv2.LINE_AA)

        return img_rgb, log_messages
    else:
        log_messages.append("No objects were segmented in the image.")
        return img_rgb, log_messages


# --- Streamlit UI ---
st.sidebar.title("App Mode")
app_mode = st.sidebar.selectbox("Choose the app mode",
    ["About", "Dimension Measurement from Image", "PPE Detection on Video"]
)

if app_mode == "About":
    st.header("About this App")
    st.write("""
        This Streamlit application combines two distinct computer vision tasks into one easy-to-use interface.
        You can choose from the following modes in the sidebar:

        - **Dimension Measurement from Image**: Upload an image containing a reference object with a known size (like a phone or a coin). The application will detect objects, calculate their dimensions in centimeters, and display the annotated image.

        - **PPE Detection on Video**: Use a video file or your webcam to detect Personal Protective Equipment (PPE). The app will process the video stream in real-time and display the annotated feed.

        Use the sidebar to select a mode and follow the on-screen instructions.
    """)

elif app_mode == "Dimension Measurement from Image":
    st.header("Dimension Measurement from Image")
    
    st.sidebar.subheader("Measurement Settings")
    ref_name = st.sidebar.text_input("Reference Object Name", "cell phone")
    known_dim = st.sidebar.number_input("Reference Object's Longest Dimension (cm)", value=16.36, step=0.1)
    
    uploaded_image = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])
    
    if uploaded_image:
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
        
        with st.spinner("Analyzing image..."):
            annotated_image, logs = measure_dimensions(uploaded_image.getvalue(), known_dim, ref_name)
        
        with col2:
            if annotated_image is not None:
                st.image(annotated_image, caption="Annotated Image", use_container_width=True)
            else:
                st.error("Could not process the image.")

        with st.expander("Processing Logs"):
            for log in logs:
                st.write(log)


elif app_mode == "PPE Detection on Video":
    st.header("PPE Detection on Video")
    
    api_key = os.getenv("API_KEY")

    st.sidebar.subheader("Video Source")
    source = st.sidebar.radio("Select source", ["Webcam", "Upload a video"])

    video_reference = None
    temp_video_path = None
    if source == "Webcam":
        video_reference = 0  # Use default webcam
    else:
        uploaded_video = st.file_uploader("Upload a video file...", type=["mp4"])
        if uploaded_video:
            # Save uploaded video to a temporary file
            with NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(uploaded_video.read())
                temp_video_path = tfile.name
                video_reference = temp_video_path

    if video_reference is not None:
        # Create a placeholder for the video feed
        image_placeholder = st.empty()
        
        col1, col2 = st.columns(2)
        start_button = col1.button("Start Inference", key="start_inference")
        stop_button = col2.button("Stop Inference", key="stop_inference")

        # Define the sink for Streamlit, with smaller annotations
        label_annotator = sv.LabelAnnotator(text_scale=0.4, text_thickness=1, text_padding=3)
        box_annotator = sv.BoxAnnotator(thickness=1)

        def streamlit_sink(predictions: dict, video_frame: VideoFrame):
            labels = [p["class"] for p in predictions["predictions"]]
            detections = sv.Detections.from_inference(predictions)
            
            # Annotate the frame
            image = label_annotator.annotate(scene=video_frame.image.copy(), detections=detections, labels=labels)
            image = box_annotator.annotate(image, detections=detections)
            
            # Display the annotated image in Streamlit
            image_placeholder.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

        if start_button and not st.session_state.is_running:
            st.session_state.pipeline = InferencePipeline.init(
                model_id="ppe-detection-huwcd/1",
                video_reference=video_reference,
                on_prediction=streamlit_sink,
                api_key=api_key,
            )
            st.session_state.pipeline.start()
            st.session_state.is_running = True
            st.info("Inference started.")

        if stop_button and st.session_state.is_running:
            if st.session_state.pipeline:
                st.session_state.pipeline.terminate()
            st.session_state.is_running = False
            # Clean up temporary file if it exists
            if temp_video_path and os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            st.info("Inference stopped.")
            # Clear the image placeholder
            image_placeholder.empty() 