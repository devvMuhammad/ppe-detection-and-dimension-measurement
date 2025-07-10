# Computer Vision Suite: Project Documentation

## 1. Introduction

In this project, we have developed a Computer Vision Suite, a sophisticated yet user-friendly application designed to perform two primary functions: real-time Personal Protective Equipment (PPE) detection and accurate dimension measurement from static images. Our goal is to make powerful computer vision tools accessible through a simple interface, allowing users to leverage complex models without requiring deep technical knowledge.

## 2. Project Features

The suite is built around two core capabilities, each addressing a distinct computer vision task.

### 2.1. Dimension Measurement
This feature measures the real-world dimensions of various objects within an image, using a single object with a known size as a reference point.

**Methodology:**
The system employs a YOLOv8 segmentation model to detect and outline all objects in a user-uploaded image. To establish a scale, the user identifies a "reference object" and provides its known longest dimension in centimeters. Our algorithm calculates a pixel-to-centimeter ratio based on this reference. This scale is then applied to all other objects recognized by the model, allowing us to compute and display their dimensions accurately.

### 2.2. Personal Protective Equipment (PPE) Detection
This feature analyzes video feeds to detect the presence of Personal Protective Equipment, enhancing workplace safety monitoring.

**Methodology:**
We utilize the Roboflow Inference Pipeline, which is integrated with a specialized model trained for PPE detection. The system processes a video stream—either from a live webcam or an uploaded file—in real-time. For each frame, the model identifies various types of PPE and annotates the video feed with bounding boxes and labels, providing immediate visual feedback.

## 3. Deployment and Interface

The entire Computer Vision Suite is deployed as an interactive web application using **Streamlit**. This framework allows us to wrap the complex backend processing into an intuitive Graphical User Interface (GUI) that runs directly in a web browser. By doing so, we make the underlying models easily accessible and operable for all users, regardless of their technical background.

## 4. User Guidelines

To use the application, select the desired mode from the sidebar.

### 4.1. Dimension Measurement Mode
1.  In the sidebar, enter the **Reference Object Name** (e.g., "cell phone").
2.  Provide the object's **Longest Dimension in centimeters**.
3.  Use the main uploader to select and upload your image file.
4.  The application will display the original image alongside the annotated image with calculated dimensions.

### 4.2. PPE Detection Mode
1.  In the sidebar, choose your **Video Source** ("Webcam" or "Upload a video").
2.  If uploading, select your video file.
3.  Press the **Start Inference** button to begin the analysis.
4.  The annotated video stream will be displayed.
5.  Press the **Stop Inference** button to terminate the session.

## 5. Technologies Used

The following technologies and libraries are the core components of this project:
- **Python**: The primary programming language.
- **Streamlit**: For building and deploying the interactive web application.
- **OpenCV**: For fundamental image and video processing tasks.
- **Ultralytics YOLOv8**: The segmentation model used for object detection in the dimension measurement tool.
- **Roboflow Inference**: The platform used for the real-time PPE detection pipeline.
- **Supervision**: For annotating images and video frames with labels and boxes.
- **NumPy**: For efficient numerical operations and data manipulation. 
- **Docker**: For deployment and packaging the final applicaion