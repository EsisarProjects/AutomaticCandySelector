import os
import shutil
from pathlib import Path
import random
import csv
import hashlib
from collections import defaultdict
from PIL import Image

# Cấu hình đường dẫn
ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw_data_set"
TARGET_DIR = ROOT_DIR / "datasets" / "classification" / "raw"
METADATA_PATH = ROOT_DIR / "datasets" / "metadata"
SPLIT_DIR = METADATA_PATH / "splits"

# Tỷ lệ chia dữ liệu
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def convert_and_rename():
    if not RAW_DIR.exists():
        print(f"❌ Không tìm thấy thư mục raw tại: {RAW_DIR}")
        return

    print("=== Bắt đầu gom, trộn ngẫu nhiên, đổi tên và chia dataset ===")
    
    all_samples = []

    # 1. Thu thập toàn bộ ảnh từ các thư mục [Loại]_[Màu] trong raw_data_set
    for folder_name in sorted(os.listdir(RAW_DIR)):
        folder_path = RAW_DIR / folder_name
        if not folder_path.is_dir():
            continue

        parts = folder_name.split("_")
        if len(parts) < 2:
            print(f"⚠️ Bỏ qua thư mục không đúng chuẩn tên: {folder_name}")
            continue

        candy_type = parts[0].lower() # Ví dụ: ourson, crocodile, dragibus
        candy_color = "_".join(parts[1:]).lower() # Ví dụ: vert, bleu, rouge

        for img_file in folder_path.glob("*.*"):
            if img_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                all_samples.append({
                    "src_path": img_file,
                    "type": candy_type,
                    "couleur": candy_color
                })

    print(f"Tổng số ảnh thu thập được: {len(all_samples)}")
    if len(all_samples) == 0:
        print("❌ Không tìm thấy ảnh nào!")
        return

    # 2. Gom nhóm theo (type, couleur) để chia tỷ lệ đều đặn (Stratified)
    grouped_samples = defaultdict(list)
    for sample in all_samples:
        grouped_samples[(sample["type"], sample["couleur"])].append(sample)

    train_records = []
    val_records = []
    test_records = []

    # Random seed để cố định kết quả chia ngẫu nhiên
    random.seed(42)

    # Đếm số thứ tự cho từng loại kẹo để đặt tên file từ 0001, 0002...
    type_counters = defaultdict(int)

    for (candy_type, candy_color), samples in grouped_samples.items():
        random.shuffle(samples) # Trộn đều các góc chụp (ngửa, nghiêng)
        
        n_total = len(samples)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)
        
        train_samples = samples[:n_train]
        val_samples = samples[n_train:n_train + n_val]
        test_samples = samples[n_train + n_val:]

        def process_subset(subset_samples, split_name):
            records = []
            for s in subset_samples:
                c_type = s["type"]
                c_color = s["couleur"]
                
                # Tăng số thứ tự cho loại kẹo này
                type_counters[c_type] += 1
                seq_id = type_counters[c_type]

                # Tên file mới theo yêu cầu: [tên_loại]_[số_thứ_tự_4_chữ_số].jpg
                ext = s["src_path"].suffix.lower()
                new_filename = f"{c_type}_{seq_id:04d}{ext}"
                
                # Cấu trúc lưu: datasets/classification/raw/type/color/split_name/image.jpg
                dest_folder = TARGET_DIR / c_type / c_color / split_name
                dest_folder.mkdir(parents=True, exist_ok=True)
                dest_path = dest_folder / new_filename

                # Copy ảnh sang đích
                shutil.copy2(s["src_path"], dest_path)

                # Lấy metadata
                rel_path = dest_path.relative_to(ROOT_DIR).as_posix()
                image_id = hashlib.sha256(rel_path.encode("utf-8")).hexdigest()[:16]
                
                with Image.open(dest_path) as img:
                    width, height = img.size
                    mode = img.mode

                records.append({
                    "image_id": image_id,
                    "image_path": rel_path,
                    "type": c_type,
                    "couleur": c_color,
                    "session_id": split_name, # Tận dụng cột này để lưu phân đoạn train/val/test
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "split": split_name
                })
            return records

        train_records.extend(process_subset(train_samples, "train"))
        val_records.extend(process_subset(val_samples, "val"))
        test_records.extend(process_subset(test_samples, "test"))

    # 3. Lưu toàn bộ ra file CSV metadata tương thích hệ thống
    METADATA_PATH.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    all_records = train_records + val_records + test_records
    fields = ["image_id", "image_path", "type", "couleur", "session_id", "width", "height", "mode", "split"]

    def save_csv(filename, data):
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)

    save_csv(METADATA_PATH / "classification.csv", all_records)
    save_csv(SPLIT_DIR / "classification_split.csv", all_records)
    save_csv(SPLIT_DIR / "classification_train.csv", train_records)
    save_csv(SPLIT_DIR / "classification_val.csv", val_records)
    save_csv(SPLIT_DIR / "classification_test.csv", test_records)

    print("\n=== HOÀN TẤT XỬ LÝ DỮ LIỆU ===")
    print(f"📁 Thư mục đích: {TARGET_DIR}")
    print(f"📊 Tổng số ảnh: {len(all_records)}")
    print(f"   - Train (70%): {len(train_records)} ảnh")
    print(f"   - Val (15%):   {len(val_records)} ảnh")
    print(f"   - Test (15%):  {len(test_records)} ảnh")
    print("✅ Đã đổi tên chuẩn dạng [type]_[0001].jpg và tạo xong các file CSV!")

if __name__ == "__main__":
    convert_and_rename()
