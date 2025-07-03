from inference_sdk import InferenceHTTPClient
import cv2
import supervision as sv
import numpy as np

# Replace ROBOFLOW_API_KEY with your Roboflow API Key
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="T9lNMapOsZJDgcOlG1nI"
)

# Load the image
image_path = "images/test.jpeg"
image = cv2.imread(image_path)

# Get prediction
result = client.infer(image_path, model_id="ppe-detection-huwcd/1")

print("Raw prediction result:")
print(result)

# Convert predictions to supervision format
detections = sv.Detections.from_inference(result)

# Create annotators with smaller text
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator(
    text_scale=0.4,  # Make text smaller (default is usually around 0.6-1.0)
    text_thickness=1,  # Thinner text
    text_padding=3,  # Less padding around text
    text_position=sv.Position.TOP_LEFT  # Position labels at top-left of boxes
)

# Get class names from the predictions
class_names = []
for prediction in result['predictions']:
    class_names.append(prediction['class'])

# Create labels with class names and confidence scores
labels = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence in zip(class_names, detections.confidence)
]

# Annotate the image
annotated_image = box_annotator.annotate(image, detections)
annotated_image = label_annotator.annotate(annotated_image, detections, labels=labels)

# Save the annotated image
output_path = "images/annotated_result.jpg"
cv2.imwrite(output_path, annotated_image)

print(f"\nAnnotated image saved to: {output_path}")
print(f"Detected {len(detections)} objects")

# Display detection summary
for i, (class_name, confidence) in enumerate(zip(class_names, detections.confidence)):
    print(f"Detection {i+1}: {class_name} (confidence: {confidence:.2f})")

# Optional: Display the image (uncomment if you want to show the image)
cv2.imshow("PPE Detection Results", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# for frame_id, frame, prediction in CLIENT.infer_on_stream("video.mp4", model_id="soccer-players-5fuqs/1"):
#     # frame_id is the number of frame
#     # frame - np.ndarray with video frame
#     # prediction - prediction from the model
#     pass

# for file_path, image, prediction in CLIENT.infer_on_stream("images", model_id="ppe-detection-huwcd/1"):
#     # file_path - path to the image
#     # frame - np.ndarray with video frame
#     # prediction - prediction from the model
#     print("prediction", prediction)
#     pass