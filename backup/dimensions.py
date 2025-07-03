# First, make sure you have the necessary libraries installed.
# You can install them using pip:
# pip install ultralytics opencv-python numpy

import cv2
import numpy as np
from ultralytics import YOLO

# --- Configuration ---
# The known real-world dimension of the reference object.
# Your phone is 163.6mm tall (16.36 cm). We use the longest dimension for calibration.
KNOWN_DIMENSION_CM = 16.36
REFERENCE_OBJECT_CLASS_NAME = "cell phone"

# The script will now use your new image.
image_path = 'images/test2.png'
output_path = 'images/annotated_dimensions.jpg'
# --- End Configuration ---


# Load the YOLOv8 instance segmentation model
model = YOLO('yolov8n-seg.pt')

# Read the image
img = cv2.imread(image_path)
if img is None:
    print(f"Error: Could not read image from {image_path}")
    print("Please make sure the image exists and the path is correct.")
    exit()

# Run the model on the image
results = model(img)

pixels_per_cm = None
reference_object_found = False

# Check if there are any detection results
if results and results[0].masks is not None:
    # --- Step 1: Find the reference object and calculate pixels_per_cm ---

    class_names = results[0].names
    # Find the class ID for our reference object ("cell phone")
    try:
        ref_class_id = list(class_names.keys())[list(class_names.values()).index(REFERENCE_OBJECT_CLASS_NAME)]
    except ValueError:
        print(f"Error: Class '{REFERENCE_OBJECT_CLASS_NAME}' not found in model's classes.")
        print(f"Available classes: {list(class_names.values())}")
        exit()

    reference_mask = None
    # Associate boxes (with class IDs) with masks to find the reference object
    if results[0].boxes:
        for i, box in enumerate(results[0].boxes):
            if box.cls == ref_class_id:
                print(f"Found reference object: '{REFERENCE_OBJECT_CLASS_NAME}'")
                # The i-th box corresponds to the i-th mask
                reference_mask = results[0].masks.xy[i]
                reference_object_found = True
                break  # Use the first one found

    if not reference_object_found:
        print(f"Error: Reference object '{REFERENCE_OBJECT_CLASS_NAME}' not detected in the image.")
        exit()

    if reference_mask is not None:
        reference_contour = np.array(reference_mask, dtype=np.int32)
        ref_rect = cv2.minAreaRect(reference_contour)
        
        # Use the longer dimension of the rectangle for calibration
        ref_pixel_dimension = max(ref_rect[1])

        if ref_pixel_dimension > 0:
            pixels_per_cm = ref_pixel_dimension / KNOWN_DIMENSION_CM
            print(f"Reference object pixel dimension: {ref_pixel_dimension:.2f} px")
            print(f"Known real-world dimension: {KNOWN_DIMENSION_CM} cm")
            print(f"Calculated pixels-per-cm ratio: {pixels_per_cm:.2f}")
        else:
            print("Error: Could not determine the pixel dimension of the reference object.")
            pixels_per_cm = None

    # --- Step 2: Measure and annotate all detected objects ---
    if pixels_per_cm:
        for i, mask in enumerate(results[0].masks.xy):
            contour = np.array(mask, dtype=np.int32)
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = box.astype(int)

            current_object_class_id = int(results[0].boxes[i].cls.item())
            color = (255, 0, 0) if current_object_class_id == ref_class_id else (255, 0, 0)
            cv2.drawContours(img, [box], 0, color, 2)

            width_px, height_px = rect[1]
            width_cm = width_px / pixels_per_cm
            height_cm = height_px / pixels_per_cm

            class_name = class_names[current_object_class_id]
            dim_text = f"{class_name}: {max(width_cm, height_cm):.1f} x {min(width_cm, height_cm):.1f} cm"

            # --- Improved text rendering ---
            font_scale = 1
            font_thickness = 2
            text_color = (255, 255, 255)  # White

            (text_w, text_h), baseline = cv2.getTextSize(dim_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

            # Center the text box on the object's center
            center_x, center_y = int(rect[0][0]), int(rect[0][1])
            text_x = center_x - text_w // 2
            text_y = center_y + text_h // 2

            # Coordinates for the background rectangle
            bg_top_left = (text_x, text_y - text_h - baseline)
            bg_bottom_right = (text_x + text_w, text_y)

            # Draw the background rectangle
            cv2.rectangle(img, bg_top_left, bg_bottom_right, color, cv2.FILLED)

            # Draw the text
            cv2.putText(img, dim_text, (text_x, text_y - baseline // 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness, cv2.LINE_AA)

        cv2.imwrite(output_path, img)
        print(f"\nAnnotated image saved to {output_path}")

    else:
        print("Could not proceed with measurements.")

else:
    print("No objects were segmented in the image.")
