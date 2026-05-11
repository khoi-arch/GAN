import os
import json
import numpy as np
import pandas as pd
import joblib
import logging
import torch
from pathlib import Path
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler("preprocess_gan.log", mode='w', encoding='utf-8'), logging.StreamHandler()])

def calc_skew_kurt(data):
    """Tính toán độ xiên (skewness) và độ nhọn (kurtosis)."""
    if len(np.unique(data)) > 1:
        return float(skew(data, bias=False)), float(kurtosis(data, bias=False))
    return 0.0, 0.0

def calculate_audit_stats(raw_data, transformed_data, scaled_data):
    """Đo lường sự thay đổi của phân phối qua 3 giai đoạn xử lý."""
    skew_raw, kurt_raw = calc_skew_kurt(raw_data)
    skew_trans, kurt_trans = calc_skew_kurt(transformed_data)
    skew_scale, kurt_scale = calc_skew_kurt(scaled_data)
    
    return {
        "Phase_1_RAW": {"skewness": round(skew_raw, 4), "kurtosis": round(kurt_raw, 4)},
        "Phase_2_TRANSFORMED": {"skewness": round(skew_trans, 4), "kurtosis": round(kurt_trans, 4)},
        "Phase_3_SCALED": {"skewness": round(skew_scale, 4), "kurtosis": round(kurt_scale, 4)},
        "Shift_Raw_to_Transformed": round(abs(skew_raw - skew_trans), 4)
    }

def calculate_covariance_shift(X_real, X_processed):
    """Tính chuẩn Frobenius để đo độ lệch hiệp phương sai."""
    cov_real = np.cov(X_real, rowvar=False)
    cov_processed = np.cov(X_processed, rowvar=False)
    return float(np.linalg.norm(cov_real - cov_processed, ord='fro'))

def run_gan_preprocessor(csv_path, groups_json_path, output_dir):
    logging.info("-> Bắt đầu chạy GAN Preprocess Pipeline (Label-First & Family-Aware Logic)...")

    # 1. Nạp dữ liệu & Định danh nhãn
    df = pd.read_csv(csv_path)
    df['Binary_Label'] = df['Class'].apply(lambda x: 0 if x == 'Benign' else 1)

    # 2. Nạp file phân loại đặc trưng
    with open(groups_json_path, 'r', encoding='utf-8') as f: 
        groups_data = json.load(f)
    
    valid_cols = []
    for group_name, cols in groups_data.items():
        if group_name != "DROP": 
            valid_cols.extend(cols)

    df_malware = df[df['Binary_Label'] == 1].copy()
    df_benign = df[df['Binary_Label'] == 0].copy()
    
    # Lọc bỏ rò rỉ nhãn (Anti-Leakage)
    meta_cols = ['Class', 'Category', 'Binary_Label', 'label_L1', 'label_L2', 'label_L3']
    all_features = sorted([c for c in df.columns if c in valid_cols and c not in meta_cols])

    # 3. Định nghĩa các chiến lược Ablation
    ablation_strategies = {
        "A_none": "none", 
        "B_q99.9": 0.999, 
        "C_q99": 0.99,
        "D_family_q99": "family_0.99", 
        "E_q99.5": 0.995,
        "F_family_q99.5": "family_0.995", 
        "G_family_q99.9": "family_0.999"
    }

    os.makedirs(output_dir, exist_ok=True)

    # =========================================================================
    # VÒNG LẶP ABLATION: Chạy song song 7 chiến lược tiền xử lý
    # =========================================================================
    for strategy_name, strategy_val in ablation_strategies.items():
        logging.info(f"\n   [+] Đang xử lý phiên bản Ablation: {strategy_name}")
        
        # Khởi tạo ma trận rỗng để hứng dữ liệu
        df_mal_processed = pd.DataFrame(index=df_malware.index)
        df_ben_processed = pd.DataFrame(index=df_benign.index)
        
        artifacts = {"scalers": {}, "clipping_bounds": {}}
        audit_report = {"strategy": strategy_name, "features_audit": {}, "global_metrics": {}}
        mal_transformed_cache = {}

        for group_name, cols_in_group in groups_data.items():
            if group_name == "DROP": continue
            
            for col in cols_in_group:
                if col not in all_features: continue
                
                mal_data = df_malware[col].copy()
                ben_data = df_benign[col].copy()
                audit_meta = {"group": group_name, "scaled": False}

                # ----------------------------------------------------
                # NHÁNH 1: BINARY & RARE -> Pass-through
                # ----------------------------------------------------
                if group_name in ["BINARY", "RARE_HIGH", "RARE_MID"]:
                    df_mal_processed[col] = mal_data.values
                    df_ben_processed[col] = ben_data.values
                    mal_transformed_cache[col] = mal_data.values
                    audit_meta["transform_applied"] = "Pass-through"

                # ----------------------------------------------------
                # NHÁNH 2: NORMAL -> Chỉ Scale (Không Clip, Không SQRT)
                # ----------------------------------------------------
                elif group_name == "NORMAL":
                    scaler = StandardScaler()
                    # Fit trên Malware, transform cho cả hai
                    mal_scaled = scaler.fit_transform(mal_data.values.reshape(-1, 1)).flatten()
                    ben_scaled = scaler.transform(ben_data.values.reshape(-1, 1)).flatten()
                    
                    df_mal_processed[col] = mal_scaled
                    df_ben_processed[col] = ben_scaled
                    mal_transformed_cache[col] = mal_data.values
                    
                    artifacts["scalers"][col] = scaler
                    audit_meta["transform_applied"] = "Scaled Only"
                    audit_meta["scaled"] = True

                # ----------------------------------------------------
                # NHÁNH 3 & 4: OUTLIER & HEAVY_TAIL -> Ablation Clipping
                # ----------------------------------------------------
                elif group_name in ["OUTLIER", "HEAVY_TAIL"]:
                    
                    # 1. BENIGN CLIPPING (Khối đồng nhất 3*IQR)
                    q25_b, q75_b = np.percentile(ben_data, 25), np.percentile(ben_data, 75)
                    iqr_b = q75_b - q25_b
                    lower_b, upper_b = q25_b - 3 * iqr_b, q75_b + 3 * iqr_b
                    ben_clipped = np.clip(ben_data.values, a_min=lower_b, a_max=upper_b)

                    # 2. MALWARE CLIPPING (Rẽ nhánh theo Ablation)
                    if strategy_val == "none" or isinstance(strategy_val, float):
                        # --- Cắt theo GLOBAL ---
                        q25_m, q75_m = np.percentile(mal_data, 25), np.percentile(mal_data, 75)
                        iqr_m = q75_m - q25_m
                        lower_m = q25_m - 3 * iqr_m

                        if strategy_val == "none":
                            clip_val = "None"
                            mal_clipped = np.clip(mal_data.values, a_min=lower_m, a_max=None)
                        else:
                            clip_val = float(np.percentile(mal_data, strategy_val * 100))
                            mal_clipped = np.clip(mal_data.values, a_min=lower_m, a_max=clip_val)
                            
                        malware_bounds_record = {"lower_global": lower_m, "upper_ablation": clip_val}

                    else: 
                        # --- Cắt theo FAMILY (Tôn trọng phân phối từng họ) ---
                        percentile_req = float(strategy_val.split("_")[1])
                        
                        # Hàm phụ trợ lấy lower fence 3*IQR cho cục bộ từng family
                        def get_family_lower_fence(x):
                            q25, q75 = np.percentile(x, 25), np.percentile(x, 75)
                            return q25 - 3 * (q75 - q25)

                        # Tính Series mảng chặn dưới/trên tương ứng với từng dòng
                        family_lower = df_malware.groupby('Class')[col].transform(get_family_lower_fence)
                        family_upper = df_malware.groupby('Class')[col].transform(lambda x: np.percentile(x, percentile_req * 100))
                        
                        # Cắt linh hoạt
                        mal_clipped = np.clip(mal_data.values, a_min=family_lower.values, a_max=family_upper.values)
                        
                        # Lưu dictionary metadata cho artifacts
                        lower_m_dict = df_malware.groupby('Class')[col].apply(get_family_lower_fence).to_dict()
                        clip_val_dict = df_malware.groupby('Class')[col].apply(lambda x: np.percentile(x, percentile_req * 100)).to_dict()
                        
                        malware_bounds_record = {"lower_family": lower_m_dict, "upper_ablation": clip_val_dict}

                    artifacts["clipping_bounds"][col] = {
                        "benign_bounds": {"lower": lower_b, "upper": upper_b},
                        "malware_bounds": malware_bounds_record
                    }

                    # 3. BIẾN ĐỔI HÌNH HỌC (Chỉ cho HEAVY_TAIL)
                    if group_name == "HEAVY_TAIL":
                        mal_transformed = np.sign(mal_clipped) * np.sqrt(np.abs(mal_clipped))
                        ben_transformed = np.sign(ben_clipped) * np.sqrt(np.abs(ben_clipped))
                        audit_meta["transform_applied"] = "Clipped (Dynamic) + Signed SQRT + Scaled"
                    else: # OUTLIER
                        mal_transformed = mal_clipped
                        ben_transformed = ben_clipped
                        audit_meta["transform_applied"] = "Clipped (Dynamic) + Scaled (No SQRT)"

                    mal_transformed_cache[col] = mal_transformed

                    # 4. SCALING
                    scaler = StandardScaler()
                    mal_scaled = scaler.fit_transform(mal_transformed.reshape(-1, 1)).flatten()
                    ben_scaled = scaler.transform(ben_transformed.reshape(-1, 1)).flatten()
                    
                    df_mal_processed[col] = mal_scaled
                    df_ben_processed[col] = ben_scaled
                    artifacts["scalers"][col] = scaler
                    audit_meta["scaled"] = True

                # Tính kiểm toán (Audit) cho cột
                stats = calculate_audit_stats(df_malware[col].values, mal_transformed_cache[col], df_mal_processed[col].values)
                audit_meta.update(stats)
                audit_report["features_audit"][col] = audit_meta

        # =====================================================================
        # KIỂM TOÁN TOÀN CỤC & XUẤT XƯỞNG
        # =====================================================================
        # Đo lường Covariance Shift
        covariance_frob_norm = calculate_covariance_shift(df_malware[all_features].values, df_mal_processed[all_features].values)
        audit_report["global_metrics"]["covariance_shift_frobenius_norm"] = covariance_frob_norm

        # Khởi tạo thư mục cho chiến lược hiện tại
        strategy_dir = os.path.join(output_dir, strategy_name)
        os.makedirs(strategy_dir, exist_ok=True)
        
        # Lưu PyTorch Tensors
        torch.save(torch.FloatTensor(df_mal_processed[all_features].values), os.path.join(strategy_dir, "tensor_malware.pt"))
        torch.save(torch.FloatTensor(df_ben_processed[all_features].values), os.path.join(strategy_dir, "tensor_benign.pt"))
        
        # Lưu Metadata (Scalers, Boundaries) để Inverse Transform
        joblib.dump(artifacts, os.path.join(strategy_dir, "preprocess_artifacts.pkl"))
        
        # Lưu báo cáo Audit
        with open(os.path.join(strategy_dir, "audit_report.json"), 'w', encoding='utf-8') as f:
            json.dump(audit_report, f, indent=4, ensure_ascii=False)

    logging.info(f"✅ HOÀN TẤT! Đã sinh 7 bộ Tensor hoàn chỉnh tại thư mục: {output_dir}")


if __name__ == "__main__":
    # Tự động map đường dẫn dựa trên cấu trúc dự án (Đảm bảo chạy được ở mọi máy)
    FILE_PATH = Path(__file__).resolve()
    PROJECT_ROOT = FILE_PATH.parent.parent 
    DATA_DIR = PROJECT_ROOT / "data"
    
    run_gan_preprocessor(
        csv_path= "/home/pak/Documents/gan_bypass_idps/data_artifacts/80-20_split/train_raw.csv",
        groups_json_path= "/home/pak/Documents/gan_bypass_idps/data_artifacts/feature_groups.json",
        output_dir= "/home/pak/Documents/gan_bypass_idps/data_artifacts/gan_tensors"
    )