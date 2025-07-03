from inference_sdk import InferenceHTTPClient
import supervision as sv
import cv2

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="T9lNMapOsZJDgcOlG1nI"
)

# Load the image
image_path = "images/pexels-cottonbro-4554249.jpg"
image = cv2.imread(image_path)

result = CLIENT.infer(image_path, model_id="object-size-measuring/1")

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