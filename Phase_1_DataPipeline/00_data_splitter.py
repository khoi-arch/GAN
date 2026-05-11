import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_data(csv_path, output_dir="../data_artifacts/80-20_split"):
    print("[0/3] Đang tiến hành Chia dữ liệu (Data Splitting) - Chống Leakage...")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df = pd.read_csv(csv_path)
    
    # Chia 80/20, Stratify theo Class để cân bằng tỉ lệ Malware/Benign
    # random_state=42 để ní chạy lại 1000 lần vẫn ra kết quả giống nhau
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['Class']
    )
    
    train_path = os.path.join(output_dir, "train_raw.csv")
    test_path = os.path.join(output_dir, "test_raw.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"✅ Đã tạo tập TRAIN: {train_path} ({len(train_df)} dòng)")
    print(f"✅ Đã tạo tập TEST: {test_path} ({len(test_df)} dòng)")

if __name__ == "__main__":
    split_data('../data_artifacts/Obfuscated-MalMem2022.csv')