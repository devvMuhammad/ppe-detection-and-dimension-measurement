from inference_sdk import InferenceHTTPClient
import cv2
import supervision as sv
import numpy as np

# Replace ROBOFLOW_API_KEY with your Roboflow API Key
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="T9lNMapOsZJDgcOlG1nI"
)

# Create annotators with smaller text
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator(
    text_scale=0.4, 
    text_thickness=1,  
    text_padding=3,  
    text_position=sv.Position.TOP_LEFT  
)

# Video output settings
output_video_path = "annotated_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 20  
video_writer = None

frame_count = 0
print("Starting video stream processing...")

for frame_id, frame, prediction in CLIENT.infer_on_stream("images/best_one_426_240_30fps.mp4", model_id="ppe-detection-huwcd/1"):
    frame_count += 1
    print(f"Processing frame {frame_id}")
    
    # Convert predictions to supervision format
    detections = sv.Detections.from_inference(prediction)
    
    # Get class names from the predictions
    class_names = []
    for pred in prediction['predictions']:
        class_names.append(pred['class'])
    
    # Create labels with class names and confidence scores
    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence in zip(class_names, detections.confidence)
    ]
    
    # Annotate the frame
    annotated_frame = box_annotator.annotate(frame, detections)
    annotated_frame = label_annotator.annotate(annotated_frame, detections, labels=labels)
    
    # Initialize video writer on first frame
    if video_writer is None:
        height, width = annotated_frame.shape[:2]
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Write frame to output video
    video_writer.write(annotated_frame)
    
    # Display the frame (optional - comment out if running headless)
    cv2.imshow("PPE Detection Stream", annotated_frame)
    
    # Print detection summary for this frame
    if len(detections) > 0:
        print(f"  Frame {frame_id}: Detected {len(detections)} objects")
        for i, (class_name, confidence) in enumerate(zip(class_names, detections.confidence)):
            print(f"    Detection {i+1}: {class_name} (confidence: {confidence:.2f})")
    else:
        print(f"  Frame {frame_id}: No detections")
    
    # Break on 'q' key press (optional)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Stopping video processing...")
        break

# Clean up
if video_writer is not None:
    video_writer.release()
cv2.destroyAllWindows()

print(f"\nProcessing completed!")
print(f"Total frames processed: {frame_count}")
print(f"Annotated video saved to: {output_video_path}") 