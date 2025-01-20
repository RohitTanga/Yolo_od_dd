YOLOv8 Object Detection and Measurement Application

Introduction:

This project demonstrates the development of an object detection and measurement application using YOLOv8. The application not only detects objects in an image but also estimates their dimensions in real-world units (inches) based on bounding box calculations. It integrates advanced computer vision techniques to provide accurate and user-friendly results, making it valuable across industries like e-commerce, architecture, and inventory management.

Features:

1.Real-time object detection and classification using YOLOv8.
2.Accurate size estimation of detected objects based on bounding box dimensions.
3.User-friendly Streamlit interface for uploading images and visualizing results.
4.Annotated output with detected objects, class labels, and dimensions.
5.Scalable and lightweight application suitable for a variety of use cases.

Workflow

Pre-processing:
Upload image through the Streamlit app interface.
Convert the uploaded image to RGB format for YOLOv8 compatibility.
Object Detection:
Predict bounding boxes, class labels, and confidence scores using YOLOv8.
Post-Processing:
Draw bounding boxes and annotate detected objects on the image.
Convert bounding box dimensions to real-world units using a specified DPI value.
Output:
Display processed images with annotations and estimated dimensions.

Setup Instructions

Prerequisites
Python 3.8+
Streamlit
OpenCV
PIL (Python Imaging Library)
Ultralytics YOLOv8

Clone the repository:

git clone https://github.com/RohitTanga/Yolo_od_dd.git  
cd Yolo_od_dd  

Install dependencies:

pip install -r requirements.txt  

Start the application:

streamlit run app.py  

Dataset

This application uses the Ultralytics COCO8 dataset, a small yet versatile object detection dataset derived from the COCO train 2017 set. For more information, visit Ultralytics COCO8 Dataset.

Applications

Livestock Monitoring: Measuring animals for health and growth tracking.
E-commerce: Validating product dimensions before shipping.
Architecture: Estimating room and object dimensions for design planning.
Inventory Management: Measuring and classifying items for efficient storage and handling.












