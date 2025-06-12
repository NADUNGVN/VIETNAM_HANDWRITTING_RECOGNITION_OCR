import os
import shutil
from pathlib import Path

# Source and destination directories
src_dir = r"C:\Users\nguye\Downloads\CamScanner 2025-05-29 14.56"
dest_dir = r"E:\WORK\project\OCR\Recognition_OCR\data\a"

# Create destination directory if it doesn't exist
os.makedirs(dest_dir, exist_ok=True)

# Get all files from source directory
files = os.listdir(src_dir)

# Process each file
for file in files:
    # Check if the file is an image (you can add more extensions if needed)
    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
        # Get file name without extension
        name, ext = os.path.splitext(file)
        
        # Create new filename with 'a' suffix
        new_filename = f"{name}a{ext}"
        
        # Full paths for source and destination
        src_path = os.path.join(src_dir, file)
        dest_path = os.path.join(dest_dir, new_filename)
        
        # Copy file with new name to destination
        shutil.copy2(src_path, dest_path)
        print(f"Copied {file} to {new_filename}")

print("Processing completed!")