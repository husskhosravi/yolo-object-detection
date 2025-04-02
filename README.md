# 🔍 YOLO Object Detection

A clean, simple Python implementation for performing object detection on images using YOLOv8.


## ✨ Key Features

- 🚀 Fast and accurate object detection using YOLOv8
- 📁 Works with individual images or entire folders
- 📊 Shows detection confidence scores
- 🎚️ Customisable confidence threshold
- 🖥️ Support for CPU/GPU inference
- 📦 Minimal dependencies

## 📥 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/husskhosravi/yolo-object-detection.git
   cd yolo-object-detection
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Basic Usage

```bash
python object_detector.py --input path/to/image.jpg --output results
```

### Process an Entire Folder

```bash
python object_detector.py --input path/to/images_folder --output results
```

### Adjust Confidence Threshold

```bash
python object_detector.py --input path/to/image.jpg --conf 0.4
```

### Use a Different YOLOv8 Model

```bash
python object_detector.py --input path/to/image.jpg --model yolov8m.pt
```

### Specify GPU Device

```bash
python object_detector.py --input path/to/image.jpg --device 0
```

## ⚙️ Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input` | Path to input image or directory of images | (Required) |
| `--output` | Path to output directory | "output" |
| `--model` | YOLOv8 model path or name | "yolov8n.pt" |
| `--conf` | Confidence threshold | 0.25 |
| `--device` | Device to run inference on (e.g., 'cpu', '0' for GPU) | (Auto) |

## 📋 Output

The script will:
1. Process each image through the YOLOv8 model
2. Draw bounding boxes around detected objects with class labels
3. Save annotated images to the output directory
4. Print detection information to the console

## 📝 Example Output

```
Loading YOLOv8 model: yolov8n.pt
Detected 3 objects in street.jpg
  car: 0.89
  person: 0.76
  dog: 0.65
```

## 🔮 Future Improvements

- [ ] Add video processing capabilities
- [ ] Implement real-time webcam detection
- [ ] Add tracking functionality for video streams
- [ ] Create a simple GUI interface
- [ ] Add export options for detection results (CSV, JSON)
- [ ] Implement batch processing with multiprocessing
- [ ] Add support for custom trained models

## 🛠️ Requirements

- Python 3.8+
- ultralytics
- OpenCV
- NumPy