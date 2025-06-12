import pandas as pd
import openpyxl
import os
import shutil
from openpyxl_image_loader import SheetImageLoader

def prepare_training_data(excel_path, output_dir):
    # Load workbook for image extraction
    wb = openpyxl.load_workbook(excel_path)
    sheet = wb.active
    image_loader = SheetImageLoader(sheet)
    
    # Read text data with pandas
    df = pd.read_excel(excel_path)
    
    # Sort DataFrame by STT to ensure correct order and remove any NaN rows
    df = df.dropna(subset=['STT'])
    df['STT'] = df['STT'].astype(int)
    df = df.sort_values(by='STT')
    
    print(f"Total valid rows in Excel: {len(df)}")
    print("First few rows after sorting:")
    print(df.head())
    
    # Create output directories
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Create labels.txt file
    labels_path = os.path.join(output_dir, 'labels.txt')
    
    processed_count = 0
    skipped_count = 0
    
    with open(labels_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            try:
                stt = int(row['STT'])
                cell = f'B{stt+1}'  # Image is in column B
                text_content = str(row['Extracted Text']).strip()
                
                if image_loader.image_in(cell):
                    image = image_loader.get(cell)
                    image_filename = f'image_0_{stt}.png'
                    image_path = os.path.join(images_dir, image_filename)
                    
                    # Save image
                    image.save(image_path)
                    
                    # Write to labels file using actual STT
                    f.write(f"images/{image_filename}\t{text_content}\n")
                    processed_count += 1
                    print(f"Processed STT {stt}: {text_content}")
                else:
                    print(f"Warning: No image found in cell {cell} for STT {stt}")
                    skipped_count += 1
                    
            except Exception as e:
                print(f"Error processing STT {stt}: {str(e)}")
                skipped_count += 1

    print(f"\nProcessing Summary:")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped rows: {skipped_count}")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    excel_path = r"E:\WORK\project\OCR\Recognition_OCR\make_label\output_excel_RNN\0.xlsx"
    output_dir = r"E:\WORK\project\OCR\Recognition_OCR\make_label\training_data"
    
    # Clear output directory first
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    prepare_training_data(excel_path, output_dir)