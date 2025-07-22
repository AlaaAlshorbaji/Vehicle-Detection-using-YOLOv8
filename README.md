# 🚗 Vehicle Detection using YOLOv8

This project demonstrates how to detect vehicles in images using the YOLOv8 (You Only Look Once, version 8) object detection algorithm developed by Ultralytics. YOLO is a state-of-the-art, real-time object detection system known for its speed and accuracy.

The notebook covers the full pipeline — from model loading to running inference and visualizing predictions on test images.

## 🎯 Objective

- Load a pretrained YOLOv8 model
- Detect vehicles in input images
- Visualize bounding boxes and class labels
- Evaluate detection results qualitatively

## 📦 Dataset

This project uses a small set of test images for inference. The dataset consists of:
- Real-world vehicle images (cars, trucks, buses, etc.)
- Pretrained YOLOv8 model trained on COCO dataset (which includes vehicle classes)

> No custom training is performed in this notebook.
## 📁 File Structure

This project is implemented entirely in the notebook:

📄 Vehicle_Detection_using_YOLOv8.ipynb


The notebook is structured as follows:

---

### 1. 📦 Installing & Importing Dependencies

- Installs the Ultralytics library:

!pip install ultralytics
Imports necessary modules:

YOLO from ultralytics

os, cv2, matplotlib, IPython.display

2. 🧠 Loading the YOLOv8 Model
Loads the pretrained model:
رmodel = YOLO('yolov8n.pt')
model = YOLO('yolov8n.pt')
The yolov8n.pt model is a lightweight version trained on the COCO dataset.

3. 🖼️ Uploading & Preparing Test Images
Uploads test images using Colab’s file interface.

Displays original images using matplotlib.

4. 🕵️ Vehicle Detection Inference
Runs the detection model on each uploaded image:
results = model.predict(source=image_path, save=True, conf=0.3)
Parameters like confidence threshold are set to control detection sensitivity.

Prediction results include bounding boxes, labels, and class confidence.

5. 📷 Visualizing Results
Displays images with predicted bounding boxes and class labels.

Saves results in a runs/detect/predict folder.

Optionally renders images inline in the notebook.

This notebook provides an easy-to-use interface for performing inference with YOLOv8 on vehicle images without needing to train a model.
## 👨‍💻 Author

**Alaa Shorbaji**  
Artificial Intelligence Instructor  
Computer Vision & Real-Time Detection Specialist  

---

## 📜 License

This project is licensed under the **MIT License**.

You are free to:

- ✅ Use and adapt the code for educational, academic, or commercial purposes  
- ✅ Modify and redistribute with appropriate credit

You must:

- ❗ Attribute the original author properly  
- ❗ Preserve this license notice in all copies or significant portions of the project

> **Disclaimer:** This project uses the official `YOLOv8` model from [Ultralytics](https://github.com/ultralytics/ultralytics), trained on the public COCO
