import pandas as pd
import numpy as np
import json
import os
import warnings

warnings.filterwarnings('ignore')

def extract_stats(csv_path, out_json="../data_artifacts/feature_stats.json"):
    print("[1/3] Đang cày CSV để trích xuất Thống kê Thô (Stats Extractor)...")
    
    df = pd.read_csv(csv_path)
    if 'Category' in df.columns:
        df = df.drop(columns=['Category'])
    df['Class'] = df['Class'].replace({'CoolWebSearch': 'CWS'})
    df['Binary_Label'] = df['Class'].apply(lambda x: 0 if x == 'Benign' else 1)
    
    features = [c for c in df.columns if c not in ['Class', 'Binary_Label']]
    total_rows = len(df)
    
    stats_dict = {}
    
    for col in features:
        col_data = df[col].dropna()
        
        # Các chỉ số cơ bản
        q01 = np.percentile(col_data, 1)
        q25 = np.percentile(col_data, 25)
        q50 = np.percentile(col_data, 50)
        q75 = np.percentile(col_data, 75)
        q99 = np.percentile(col_data, 99)
        
        unique_vals = len(np.unique(col_data))
        zero_ratio = (col_data == 0).sum() / total_rows
        
        # --- FIX LOGIC TẠI ĐÂY ---
        # Chỉ lấy index của những dòng mà giá trị feature > 0
        active_indices = col_data[col_data > 0].index
        active_count = len(active_indices)
        
        # Chỉ đếm Malware trong tập active_indices đó
        if active_count > 0:
            malware_count = df.loc[active_indices, 'Binary_Label'].sum()
        else:
            malware_count = 0
        # -------------------------
        
        stats_dict[col] = {
            "quantiles": {"1%": float(q01), "25%": float(q25), "50%": float(q50), "75%": float(q75), "99%": float(q99), "100%": float(col_data.max())},
            "max": float(col_data.max()),
            "std": float(col_data.std()),
            "mean": float(col_data.mean()), # Thêm mean để dùng cho logic Outlier
            "unique_values": int(unique_vals),
            "zero_ratio": float(zero_ratio),
            "skewness": float(col_data.skew()),
            "kurtosis": float(col_data.kurtosis()),
            "active_count": int(active_count),
            "malware_count": int(malware_count)
        }

    output = {
        "_summary": {"total_rows_analyzed": total_rows},
        "features": stats_dict
    }

    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
        
    print(f"✅ Hoàn tất tính toán! Đã lưu: {out_json}")

if __name__ == "__main__":
    extract_stats('/home/pak/Documents/gan_bypass_idps/data_artifacts/80-20_split/train_raw.csv') # Nhớ check đúng path của bạn