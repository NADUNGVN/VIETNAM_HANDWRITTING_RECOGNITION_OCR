from ultralytics import YOLO
import cv2
import time
import torch
import os
import json
import concurrent.futures
from pathlib import Path
import logging

# Đường dẫn đến mô hình và hình ảnh
model_path = r"E:\WORK\project\OCR\Recognition_OCR\model\best_yolo12n_v2_5_6_314img.pt"
image_path = r"E:\WORK\project\OCR\Recognition_OCR\data\test\19.png"
output_dir = r"E:\WORK\project\OCR\Recognition_OCR\data_results\imgs"
result_dir = r"E:\WORK\project\OCR\Recognition_OCR\data_results\result_images"

# Tạo thư mục kết quả và thư mục cho result images
for dir_path in [output_dir, result_dir]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Tối ưu hóa cho CPU - sử dụng tất cả các luồng có sẵn
torch.set_num_threads(6)  # Sử dụng tất cả 8 luồng của CPU i5-8265U

# Tải mô hình
print(f"Đang tải mô hình từ {model_path}...")
try:
    model = YOLO(model_path)
    print("Đã tải mô hình thành công!")
except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}")
    exit(1)

# Chuyển mô hình sang CPU
model.to('cpu')

# Đọc hình ảnh
print(f"Đang đọc hình ảnh từ {image_path}...")
try:
    img = cv2.imread(image_path)
    if img is None:
        print(f"Không thể đọc hình ảnh từ {image_path}")
        exit(1)
    print(f"Đã đọc hình ảnh thành công! Kích thước: {img.shape}")
except Exception as e:
    print(f"Lỗi khi đọc hình ảnh: {e}")
    exit(1)

# Thực hiện dự đoán
print("Đang xử lý hình ảnh...")
start_time = time.time()

# Thực hiện dự đoán với ngưỡng tin cậy 0.25 (có thể điều chỉnh)
results = model(img, conf=0.3, iou=0.45, device='cpu')
if results is None or len(results) == 0:
    print("Không phát hiện được đối tượng nào trong hình ảnh.")
    exit(1)

end_time = time.time()
processing_time = (end_time - start_time) * 1000  

# Vẽ và lưu ảnh với bounding boxes
result_img = results[0].plot()  # Ảnh với labels và boxes
result_img_boxes_only = img.copy()  # Tạo bản copy cho ảnh chỉ có boxes

# Vẽ chỉ boxes không có labels
boxes = results[0].boxes
for box in boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    cv2.rectangle(result_img_boxes_only, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Lưu cả 2 loại ảnh
cv2.imwrite(os.path.join(result_dir, "result_with_labels.png"), result_img)
cv2.imwrite(os.path.join(result_dir, "result_boxes_only.png"), result_img_boxes_only)

boxes = results[0].boxes
detected_objects = []

# Hàm kiểm tra tính hợp lệ của bounding box
def validate_box(box, img_shape):
    x1, y1, x2, y2 = box
    height, width = img_shape[:2]
    
    # Kiểm tra tọa độ có hợp lệ không
    if x1 >= x2 or y1 >= y2:
        return False
    
    # Kiểm tra box có nằm trong ảnh không
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        return False
    
    # Kiểm tra kích thước tối thiểu
    if (x2 - x1) < 5 or (y2 - y1) < 5:
        return False
        
    return True

# Tạo danh sách các đối tượng với thông tin vị trí
for i, box in enumerate(boxes):
    class_id = int(box.cls[0].item())
    class_name = model.names[class_id]
    confidence = box.conf[0].item()
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    
    # Kiểm tra tính hợp lệ của box
    if not validate_box((x1, y1, x2, y2), img.shape):
        print(f"Bỏ qua box không hợp lệ: {(x1, y1, x2, y2)}")
        continue
    
    # Thêm vào danh sách với tọa độ center để sắp xếp
    center_y = (y1 + y2) // 2
    center_x = (x1 + x2) // 2
    detected_objects.append({
        'index': i,
        'class_name': class_name,
        'confidence': confidence,
        'coords': (x1, y1, x2, y2),
        'center_y': center_y,
        'center_x': center_x
    })

# Sắp xếp đối tượng theo dòng (row-first)
# Định nghĩa ngưỡng y để xác định cùng dòng
y_threshold = 30  # Có thể điều chỉnh giá trị này

# Sắp xếp theo y trước
detected_objects.sort(key=lambda x: x['center_y'])

# Gom nhóm các đối tượng cùng dòng và sắp xếp từ trái sang phải
current_y = detected_objects[0]['center_y']
current_line = []
final_sorted_objects = []

for obj in detected_objects:
    if abs(obj['center_y'] - current_y) <= y_threshold:
        current_line.append(obj)
    else:
        # Sắp xếp đối tượng trong dòng hiện tại từ trái sang phải
        current_line.sort(key=lambda x: x['center_x'])
        final_sorted_objects.extend(current_line)
        current_line = [obj]
        current_y = obj['center_y']

# Xử lý dòng cuối cùng
if current_line:
    current_line.sort(key=lambda x: x['center_x'])
    final_sorted_objects.extend(current_line)

# In kết quả và lưu ảnh theo thứ tự mới
print(f"\nKết quả phân tích OCR (đã sắp xếp theo thứ tự đọc):")
print(f"- Thời gian xử lý: {processing_time:.2f}ms")
print(f"- Số đối tượng phát hiện được: {len(final_sorted_objects)}")

if len(final_sorted_objects) > 0:
    print("\nChi tiết các đối tượng:")
    for i, obj in enumerate(final_sorted_objects):
        x1, y1, x2, y2 = obj['coords']
        
        # Cắt vùng ảnh từ ảnh gốc
        cropped_img = img[y1:y2, x1:x2]
        
        # Lưu ảnh đã cắt
        crop_filename = os.path.join(output_dir, f"crop_{i+1:03d}_{obj['class_name']}.png")
        cv2.imwrite(crop_filename, cropped_img)
        
        print(f"  {i+1}. {obj['class_name']}: {obj['confidence']*100:.1f}% - Vị trí: [{x1}, {y1}, {x2}, {y2}]")
        print(f"     Đã lưu ảnh cắt tại: {crop_filename}")

# Lưu thông tin các đối tượng đã sắp xếp vào file JSON
processed_results = []
for i, obj in enumerate(final_sorted_objects):
    x1, y1, x2, y2 = obj['coords']
    
    processed_results.append({
        'index': i + 1,  # Đánh số từ 1
        'class_name': obj['class_name'],
        'confidence': float(obj['confidence']),  # Chuyển đổi numpy float sang Python float
        'coords': [int(x1), int(y1), int(x2), int(y2)],  # Chuyển đổi sang list để có thể serialize
        'center_y': int(obj['center_y']),
        'center_x': int(obj['center_x'])
    })

# Lưu kết quả vào file JSON
json_output_path = os.path.join(output_dir, "processed_results.json")
try:
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_results, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu thông tin đối tượng vào: {json_output_path}")
except Exception as e:
    print(f"Lỗi khi lưu file JSON: {str(e)}")

print("\nQuá trình xử lý hoàn tất!")

# Thêm hàm validation paths
def validate_paths():
    paths = {
        'model': Path(model_path),
        'image': Path(image_path),
        'output': Path(output_dir)
    }
    
    for name, path in paths.items():
        if name in ['model', 'image']:
            if not path.exists():
                raise FileNotFoundError(f"{name} path không tồn tại: {path}")
        if name == 'output':
            path.mkdir(parents=True, exist_ok=True)

# Hàm xử lý và lưu ảnh đã cắt
def process_and_save_crop(args):
    i, obj, img, total = args
    x1, y1, x2, y2 = obj['coords']
    cropped_img = img[y1:y2, x1:x2]
    crop_filename = os.path.join(output_dir, f"crop_{i+1:03d}_{obj['class_name']}.png")
    cv2.imwrite(crop_filename, cropped_img)
    
    # Hiển thị tiến trình xử lý
    progress = ((i + 1) / total) * 100
    print(f"\rXử lý ảnh crop: {progress:.1f}% ({i + 1}/{total})", end="", flush=True)
    
    return crop_filename

def main():
    try:
        validate_paths()
        
        total_objects = len(final_sorted_objects)
        print(f"\nBắt đầu xử lý {total_objects} đối tượng...")
        
        # Thêm total vào tham số
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            crop_tasks = [(i, obj, img, total_objects) for i, obj in enumerate(final_sorted_objects)]
            results = list(executor.map(process_and_save_crop, crop_tasks))
        
        print("\nHoàn thành cắt và lưu ảnh!")
        print(f"\nKết quả xử lý:")
        print(f"- Ảnh với nhãn và boxes: {os.path.join(result_dir, 'result_with_labels.png')}")
        print(f"- Ảnh chỉ có boxes: {os.path.join(result_dir, 'result_boxes_only.png')}")
        
        for i, filename in enumerate(results):
            obj = final_sorted_objects[i]
            print(f"  {i+1}. {obj['class_name']}: {obj['confidence']*100:.1f}%")
            
    except Exception as e:
        logging.error(f"Lỗi: {str(e)}")
        raise

if __name__ == "__main__":
    main()

