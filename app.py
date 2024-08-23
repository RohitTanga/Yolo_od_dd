import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load YOLO model
yolo_model = YOLO('yolov8s.pt')

def detect_and_measure(uploaded_image):
    # Convert the uploaded image to RGB
    image_np = np.array(uploaded_image.convert('RGB'))

    # Run inference on the uploaded image
    detection_results = yolo_model(image_np, imgsz=640, conf=0.5)

    # Extract bounding box dimensions and labels
    bounding_boxes = detection_results[0].boxes.xywh.cpu().numpy()  # Bounding boxes in xywh format
    class_labels = detection_results[0].names  # Class labels
    class_ids = detection_results[0].boxes.cls.cpu().numpy().astype(int)  # Class IDs

    # Convert image to PIL format for matplotlib
    image_pil = Image.fromarray(image_np)

    # Display the image with bounding boxes
    fig, ax = plt.subplots()
    ax.imshow(image_pil)
    
    dpi = 96  # Dots per inch for conversion
    for box, class_id in zip(bounding_boxes, class_ids):
        x_center, y_center, box_width, box_height = box
        x1, y1 = int(x_center - box_width / 2), int(y_center - box_height / 2)
        x2, y2 = int(x_center + box_width / 2), int(y_center + box_height / 2)
        
        # Draw bounding box
        rect = plt.Rectangle((x1, y1), box_width, box_height, fill=False, color='red', linewidth=2)
        ax.add_patch(rect)
        
        # Convert dimensions from pixels to inches
        width_in_inches = box_width / dpi
        height_in_inches = box_height / dpi
        
        # Get the label of the detected object
        object_label = class_labels[class_id]
        
        # Print dimensions and label
        st.write(f"Object: {object_label}, Width of Box: {width_in_inches:.2f} inches, Height of Box: {height_in_inches:.2f} inches")
        
        # Annotate image with dimensions and label
        plt.text(x1, y1 - 10, f'{object_label}: {width_in_inches:.2f} inches x {height_in_inches:.2f} inches', 
                 color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    plt.axis('off')
    st.pyplot(fig)

def main():
    st.title('YOLOv8 Object Detection and Measurement App')

    # File uploader for image with multiple formats
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "heic", "bmp", "gif"])
    if uploaded_file is not None:
        # Load image with PIL
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Detecting objects and measuring...")
        detect_and_measure(image)

if __name__ == "__main__":
    main()
