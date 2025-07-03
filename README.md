# Computer Vision Suite

This is a Streamlit application that provides a suite of computer vision tools.

## Features

- **Dimension Measurement from Image**: Upload an image with a reference object to measure the dimensions of other objects in the image.
- **PPE Detection on Video**: Analyze a video stream from a file or webcam to detect Personal Protective Equipment (PPE).

## Getting Started

### Prerequisites

- Python 3.8+
- An API key from Roboflow for PPE detection.

### Installation & Running

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd ppe-detection
    ```

2.  **Create a `.env` file:**
    Create a `.env` file in the root of the project and add your Roboflow API key:
    ```
    API_KEY="YOUR_API_KEY"
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    streamlit run app.py
    ``` 