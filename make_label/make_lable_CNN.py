import torch
from PIL import Image
import cv2
import os
from ultralytics import YOLO

# Load YOLO model
model = YOLO('E:/WORK/project/OCR/Recognition_OCR/model/best_yolo12nv1_26_5.pt')

# Add class names dictionary
class_names = {
    0: 'simple_handwritten',
    1: 'special_character'
}

def process_image(image_path, output_dir, output_label_dir):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return
    
    # Make a copy for labeling
    labeled_img = img.copy()
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Create output paths
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    label_path = os.path.join(output_dir, f"{base_name}.txt")
    orig_img_path = os.path.join(output_dir, f"{base_name}.png")
    bbox_only_path = os.path.join(output_label_dir, f"{base_name}.png")
    
    # Create a separate copy for bbox-only image
    bbox_only_img = img.copy()
    
    # Run inference
    results = model(img)
    
    # Open label file
    with open(label_path, 'w') as f:
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                
                # Convert to YOLO format
                x_center = ((x1 + x2) / 2) / width
                y_center = ((y1 + y2) / 2) / height
                w = (x2 - x1) / width
                h = (y2 - y1) / height
                
                cls = int(box.cls[0])
                cls_name = class_names[cls]
                
                # Write to label file
                f.write(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
                
                # Draw on the labeled image copy
                cv2.rectangle(labeled_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"{cls_name}: {float(box.conf[0]):.2f}"
                cv2.putText(labeled_img, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw only bounding box without class name on bbox_only_img
                cv2.rectangle(bbox_only_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    # Save both original and labeled images
    cv2.imwrite(orig_img_path, img)  
    cv2.imwrite(bbox_only_path, bbox_only_img)
    print(f"Saved original image, labeled image, and label file for: {base_name}")

def main():
    # Define input and output directories
    input_dir = r"E:/WORK/project/OCR/Recognition_OCR/data/a"
    output_dir = r"E:/WORK/project/OCR/Recognition_OCR/make_label/output/a"
    output_label_dir = r"E:/WORK/project/OCR/Recognition_OCR/make_label/output_label/a"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # Process all images in directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            process_image(input_path, output_dir, output_label_dir)  # Added output_label_dir parameter
            print(f"Processing: {filename}")
        else:
            continue

if __name__ == "__main__":
    main()
