from inference import get_model
import supervision as sv
import cv2

# define the image url to use for inference
image_file = "images/test.jpeg"
image = cv2.imread(image_file)

# load a pre-trained yolov8n model
model = get_model(model_id="ppe-detection-huwcd/1")

# run inference on our chosen image, image can be a url, a numpy array, a PIL image, etc.
results = model.infer(image)

print(results)

# load the results into the supervision Detections api
# detections = sv.Detections.from_inference(results)

# # create supervision annotators
# bounding_box_annotator = sv.BoxAnnotator()
# label_annotator = sv.LabelAnnotator()

# # annotate the image with our inference results
# annotated_image = bounding_box_annotator.annotate(
#     scene=image, detections=detections)
# annotated_image = label_annotator.annotate(
#     scene=annotated_image, detections=detections)

# # display the image
# sv.show_image(annotated_image)