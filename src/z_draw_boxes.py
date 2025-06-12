import cv2
import json
import numpy as np

# Đường dẫn files
json_path = r"E:\WORK\project\OCR\Recognition_OCR\data_results\ocr_output\json_results.json"
img_path = r"E:\WORK\project\OCR\Recognition_OCR\data\test\19.png"
output_path = r"E:\WORK\project\OCR\Recognition_OCR\data_results\visualization.png"

# Đọc ảnh và JSON
img = cv2.imread(img_path)
with open(json_path, 'r', encoding='utf-8') as f:
    results = json.load(f)

# Vẽ boxes
for item in results:
    if item.get('original_crop_coords'):
        # Lấy tọa độ
        x1, y1, x2, y2 = item['original_crop_coords']
        text = item.get('full_text', '')
        
        # Tạo overlay cho box
        overlay = img.copy()
        
        # Chọn màu dựa vào có text hay không
        if text.strip():  # Nếu có text
            fill_color = (144, 238, 144)  # Màu xanh nhạt
            border_color = (0, 255, 0)    # Viền xanh lá
        else:  # Nếu không có text
            fill_color = (71, 130, 255)   # Màu cam nhạt 
            border_color = (0, 165, 255)  # Viền cam đậm
        
        # Tô màu box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), fill_color, -1)
        
        # Áp dụng độ trong suốt
        alpha = 0.4
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        
        # Vẽ viền box
        cv2.rectangle(img, (x1, y1), (x2, y2), border_color, 2)
        
        # Chỉ vẽ text nếu có
        if text.strip():
            # Tính vị trí text ở giữa box
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6  # Tăng kích thước font
            thickness = 2     # Tăng độ đậm của text
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            
            text_x = x1 + (x2 - x1 - text_width) // 2
            text_y = y1 + (y2 - y1 + text_height) // 2
            
            # Vẽ text màu đen đậm
            cv2.putText(img, text, (text_x, text_y), font, 
                       font_scale, (0, 0, 0), thickness)

# Lưu ảnh kết quả        
cv2.imwrite(output_path, img)
print(f"Đã lưu ảnh kết quả tại: {output_path}")
