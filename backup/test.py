import cv2
import numpy as np
from ultralytics import YOLO

image_path = 'images/test2.png'
output_path = 'shit.jpg'

# Load the YOLOv8 instance segmentation model
model = YOLO('yolov8n-seg.pt')
# Read the image
img = cv2.imread(image_path)
if img is None:
    print(f"Error: Could not read image from {image_path}")
    print("Please make sure the image exists and the path is correct.")
    exit()

# Run the model on the image with lower confidence threshold to detect more objects
results = model(img)

# Print detection details
for i, result in enumerate(results):
    print(f"Result {i}:")
    print(f"Number of detections: {len(result.boxes)}")
    if result.boxes is not None:
        for j, box in enumerate(result.boxes):
            print(f"  Detection {j}: class={int(box.cls)}, confidence={float(box.conf):.3f}")

# annotate the image with all results
annotated_image = results[0].plot()

# save the annotated image
cv2.imwrite(output_path, annotated_image)

# print the results
print(results)