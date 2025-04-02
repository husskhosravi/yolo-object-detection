#!/usr/bin/env python3
"""
YOLOv8 Object Detection Script
A clean, simple implementation for performing object detection on images.
"""

import os
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="YOLOv8 Object Detection")
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Path to input image or directory of images"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="output", 
        help="Path to output directory"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="yolov8n.pt", 
        help="YOLOv8 model path or name"
    )
    parser.add_argument(
        "--conf", 
        type=float, 
        default=0.25, 
        help="Confidence threshold"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="", 
        help="Device to run inference on (e.g., 'cpu', '0' for GPU)"
    )
    return parser.parse_args()


def process_image(model, image_path, output_dir, conf_threshold):
    """Process a single image with the YOLOv8 model."""
    # Read the image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return

    # Get image name for saving
    image_name = os.path.basename(image_path)
    
    # Run YOLOv8 inference
    results = model(image, conf=conf_threshold)
    
    # Visualise the results
    annotated_image = results[0].plot()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the annotated image
    output_path = os.path.join(output_dir, f"detected_{image_name}")
    cv2.imwrite(output_path, annotated_image)
    
    # Print detection information
    boxes = results[0].boxes
    print(f"Detected {len(boxes)} objects in {image_name}")
    
    # Print list of detected objects with confidence scores
    for box in boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        print(f"  {class_name}: {confidence:.2f}")
    
    return output_path


def main():
    """Main function to run object detection."""
    # Parse arguments
    args = parse_arguments()
    
    # Load the YOLOv8 model
    print(f"Loading YOLOv8 model: {args.model}")
    model = YOLO(args.model)
    
    # Set device if specified
    if args.device:
        model.to(args.device)
    
    # Determine if input is a single image or directory
    input_path = Path(args.input)
    output_dir = Path(args.output)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if input_path.is_file():
        # Process a single image
        process_image(model, input_path, output_dir, args.conf)
    elif input_path.is_dir():
        # Process all images in the directory
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        image_paths = [
            f for f in input_path.iterdir() 
            if f.suffix.lower() in image_extensions
        ]
        
        if not image_paths:
            print(f"No images found in {input_path}")
            return
        
        print(f"Found {len(image_paths)} images in {input_path}")
        for img_path in image_paths:
            process_image(model, img_path, output_dir, args.conf)
    else:
        print(f"Error: {args.input} is not a valid file or directory")


if __name__ == "__main__":
    main()