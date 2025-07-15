import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
from supervision.detection.core import Detections
from supervision.annotators.core import LabelAnnotator, BoxAnnotator
import os
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv
import time

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
if 'last_toast_time' not in st.session_state:
    st.session_state.last_toast_time = 0
if 'consecutive_violations' not in st.session_state:
    st.session_state.consecutive_violations = 0

# --- Core Functions ---

def annotate_known_dimensions(image_bytes, object_name, known_height_m, known_width_m):
    """
    Processes an image to find a specific object and annotates it with known dimensions.
    """
    model = load_yolo_model()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = model(img_rgb)
    log_messages = []

    if results and results[0].masks is not None:
        class_names = results[0].names
        try:
            target_class_id = list(class_names.keys())[list(class_names.values()).index(object_name.lower())]
        except ValueError:
            log_messages.append(f"Error: Class '{object_name}' not found in model.")
            log_messages.append(f"Available classes: {list(class_names.values())}")
            return None, log_messages

        object_found = False
        if results[0].boxes:
            for i, box in enumerate(results[0].boxes):
                if box.cls == target_class_id:
                    log_messages.append(f"Found object: '{object_name}'")
                    mask = results[0].masks.xy[i]
                    object_found = True
                    
                    contour = np.array(mask, dtype=np.int32)
                    rect = cv2.minAreaRect(contour)
                    box_points = cv2.boxPoints(rect)
                    box_points = box_points.astype(int)

                    cv2.drawContours(img_rgb, [box_points], 0, (0, 255, 0), 3)

                    # Create text for dimensions
                    dim_text = f"H: {known_height_m:.2f}m, W: {known_width_m:.2f}m"
                    
                    # Get text size to position it
                    font_scale = 0.8
                    font_thickness = 2
                    (text_w, text_h), baseline = cv2.getTextSize(dim_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    
                    # Position text at the top-left corner of the bounding box
                    text_x = box_points[1][0] 
                    text_y = box_points[1][1] - text_h - 10

                    # Add a background rectangle for the text for better readability
                    cv2.rectangle(img_rgb, (text_x, text_y - text_h), (text_x + text_w, text_y + baseline), (0, 0, 0), cv2.FILLED)
                    cv2.putText(img_rgb, dim_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

                    break 

        if not object_found:
            log_messages.append(f"Error: Object '{object_name}' not detected.")
            return img_rgb, log_messages

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
    
    st.sidebar.subheader("Object Settings")
    object_name = st.sidebar.text_input("Object Name", "car")
    known_height = st.sidebar.number_input("Object Height (m)", value=1.41, step=0.01)
    known_width = st.sidebar.number_input("Object Width (m)", value=1.40, step=0.01)

    uploaded_image = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])
    
    if uploaded_image:
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
        
        with st.spinner("Analyzing image..."):
            annotated_image, logs = annotate_known_dimensions(
                uploaded_image.getvalue(), 
                object_name, 
                known_height, 
                known_width
            )
        
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

    st.sidebar.subheader("Violation Settings")
    consecutive_threshold = st.sidebar.number_input(
        "Consecutive Violation Threshold", 
        min_value=1, 
        max_value=20, 
        value=3, 
        step=1,
        help="Number of consecutive frames with violations needed to trigger alert"
    )
    
    st.sidebar.subheader("Video Settings")
    fps_limit = st.sidebar.slider(
        "Max FPS", 
        min_value=1, 
        max_value=30, 
        value=5, 
        step=1,
        help="Limit the frame rate to reduce processing speed"
    )

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
        warning_placeholder = st.empty()
        debug_placeholder = st.empty()
        image_placeholder = st.empty()
        
        col1, col2 = st.columns(2)
        start_button = col1.button("Start Inference", key="start_inference")
        stop_button = col2.button("Stop Inference", key="stop_inference")

        # Define the sink for Streamlit, with smaller annotations
        label_annotator = LabelAnnotator(text_scale=0.4, text_thickness=1, text_padding=3)
        box_annotator = BoxAnnotator(thickness=1)

        # Define violation classes based on your model's output
        VIOLATION_CLASSES = {"NO-Hardhat", "NO-Safety Vest"}

        def streamlit_sink(predictions: dict, video_frame: VideoFrame):
            labels = [p["class"] for p in predictions["predictions"]]
            detections = Detections.from_inference(predictions)
            
            # Check for violations
            violation_detected = any(label in VIOLATION_CLASSES for label in labels)

            # Handle consecutive violations
            if violation_detected:
                st.session_state.consecutive_violations += 1
            else:
                st.session_state.consecutive_violations = 0

            # Debug information
            debug_info = f"**Debug Info:**\n"
            debug_info += f"- Total detections: {len(labels)}\n"
            debug_info += f"- Detected classes: {list(set(labels))}\n"
            debug_info += f"- Violation classes: {VIOLATION_CLASSES}\n"
            debug_info += f"- Violation detected: {violation_detected}\n"
            debug_info += f"- Consecutive violations: {st.session_state.consecutive_violations}/{consecutive_threshold}\n"
            
            debug_placeholder.markdown(debug_info)

            if st.session_state.consecutive_violations >= consecutive_threshold:
                current_time = time.time()
                if current_time - st.session_state.last_toast_time > 5:  # 5 second cooldown
                    st.toast(f"🚨 PPE Violation Detected! ({st.session_state.consecutive_violations} consecutive frames)", icon="⚠️")
                    st.session_state.last_toast_time = current_time
                warning_placeholder.error(f"🚨 PPE Violation Detected! ({st.session_state.consecutive_violations} consecutive frames)")
            else:
                warning_placeholder.empty()
            
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
                max_fps=fps_limit
            )
            st.session_state.pipeline.start()
            st.session_state.is_running = True
            st.info("Inference started.")

        if stop_button and st.session_state.is_running:
            if st.session_state.pipeline:
                st.session_state.pipeline.terminate()
            st.session_state.is_running = False
            st.session_state.consecutive_violations = 0  # Reset counter
            # Clean up temporary file if it exists
            if temp_video_path and os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            st.info("Inference stopped.")
            # Clear the placeholders
            warning_placeholder.empty()
            debug_placeholder.empty()
            image_placeholder.empty() 