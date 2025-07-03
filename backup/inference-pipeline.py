# Import the InferencePipeline object
from inference import InferencePipeline
# Import the built in render_boxes sink for visualizing results
from inference.core.interfaces.camera.entities import VideoFrame
import supervision as sv
import cv2

label_annotator = sv.LabelAnnotator()
box_annotator = sv.BoxAnnotator()

def my_custom_sink(predictions: dict, video_frame: VideoFrame):
    # get the text labels for each prediction
    labels = [p["class"] for p in predictions["predictions"]]
    # load our predictions into the Supervision Detections api
    detections = sv.Detections.from_inference(predictions)
    # annotate the frame using our supervision annotator, the video_frame, the predictions (as supervision Detections), and the prediction labels
    image = label_annotator.annotate(
        scene=video_frame.image.copy(), detections=detections, labels=labels
    )
    image = box_annotator.annotate(image, detections=detections)
    # display the annotated image
    cv2.imshow("Predictions", image)
    cv2.waitKey(1)

# initialize a pipeline object
pipeline = InferencePipeline.init(
    model_id="ppe-detection-huwcd/1", # Roboflow model to use
    video_reference="images/best_one_426_240_30fps.mp4", # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
    on_prediction=my_custom_sink, # Function to run after each prediction
    api_key="T9lNMapOsZJDgcOlG1nI",

)

pipeline.start()
pipeline.join()