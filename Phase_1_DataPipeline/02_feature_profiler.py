import numpy as np
import json
import math
import os

EPS = 1e-6

class RareFeatureScorer:
    def __init__(self, alpha=2.0, base_rate=0.5):
        self.alpha = alpha
        self.base_rate = base_rate

    def score_feature(self, col, stats_dict):
        m = float(stats_dict.get("malware_count", 0))
        n = float(stats_dict.get("active_count", 0))
        
        if n == 0: return "DROP_RARE_LOW_SCORE", 0.0

        mean = (m + self.alpha) / (n + 2 * self.alpha)
        var = (mean * (1.0 - mean)) / (n + 2 * self.alpha + 1.0)
        std_dev = math.sqrt(var)
        
        score = abs(mean - self.base_rate) / (std_dev + 1e-9) 

        if score >= 2.5: action = "KEEP_RARE_HIGH_SCORE"
        elif score >= 1.5: action = "TRANSFORM_RARE_MID_SCORE"
        else: action = "DROP_RARE_LOW_SCORE"

        return action, score

def assign_action(s, total_rows):
    # Lấy thống kê cơ bản
    q50 = s.get("quantiles", {}).get("50%", 0.0)
    q99 = s.get("quantiles", {}).get("99%", 0.0)
    unique_vals = s.get("unique_values", 0)
    zero_ratio = s.get("zero_ratio", 0.0)
    skew = s.get("skewness", 0.0)
    kurt = s.get("kurtosis", 0.0)
    
    mean = s.get("mean", 0.0)
    std = s.get("std", 0.0)
    max_val = s.get("max", 0.0)

    # Lọc hằng số và cờ nhị phân
    if unique_vals < 2: return "DROP_CONSTANT"
    if unique_vals == 2:
        minor_class_ratio = min(zero_ratio, 1.0 - zero_ratio)
        if minor_class_ratio > 0.001: return "KEEP_RAW_BINARY"

    # Lọc biến hiếm (Rare)
    rare_ratio = 1.0 - zero_ratio
    rare_count = total_rows * rare_ratio
    if rare_count < 50 or rare_ratio < 0.005: return "RARE_NEED_SCORING"

    # --- LOGIC CỦA BẠN: Tính toán Outlier Ratios ---
    median_safe = q50 if abs(q50) > EPS else 1e-9
    outlier_ratio_median = max_val / median_safe
    outlier_ratio_std = (max_val - mean) / (std + 1e-9)

    # 1. Check Outlier
    if (abs(median_safe) > EPS and outlier_ratio_median > 20 and outlier_ratio_std > 5) or \
       (abs(q99) > EPS and (max_val / q99) > 10):
        return "TRANSFORM_OUTLIER"

    # 2. Check Tail
    if kurt > 5.0: return "TRANSFORM_HEAVY_TAIL"
    if skew > 1.0: return "TRANSFORM_LONG_TAIL_RIGHT"
    elif skew < -1.0: return "TRANSFORM_LONG_TAIL_LEFT"
    
    # 3. Mặc định
    return "KEEP_RAW_NORMAL"

def run_profiler(stats_json_path="/home/pak/Documents/gan_bypass_idps/data_artifacts/feature_stats.json", groups_json_path="/home/pak/Documents/gan_bypass_idps/data_artifacts/feature_groups.json"):
    print("[2/3] Chạy Logic Gom nhóm Đặc trưng (Feature Profiler)...")
    
    if not os.path.exists(stats_json_path):
        print(f"❌ Lỗi: Cần chạy 01_stats_extractor.py trước!")
        return

    with open(stats_json_path, 'r', encoding='utf-8') as f: 
        full_data = json.load(f)
        
    stats_dict = full_data.get("features", {})
    total_rows = full_data.get("_summary", {}).get("total_rows_analyzed", 0)

    rare_scorer = RareFeatureScorer(alpha=2.0, base_rate=0.5)
    
    groups = {
        "BINARY": [], "RARE_HIGH": [], "RARE_MID": [], 
        "OUTLIER": [], "HEAVY_TAIL": [], "NORMAL": [], "DROP": []
    }

    for col, s in stats_dict.items():
        action = assign_action(s, total_rows)
        
        if action == "RARE_NEED_SCORING":
            final_action, _ = rare_scorer.score_feature(col, s)
            action = final_action

        if action.startswith("DROP"): groups["DROP"].append(col)
        elif action == "KEEP_RAW_BINARY": groups["BINARY"].append(col)
        elif action == "KEEP_RARE_HIGH_SCORE": groups["RARE_HIGH"].append(col)
        elif action == "TRANSFORM_RARE_MID_SCORE": groups["RARE_MID"].append(col)
        elif action == "TRANSFORM_OUTLIER": groups["OUTLIER"].append(col)
        elif action in ["TRANSFORM_HEAVY_TAIL", "TRANSFORM_LONG_TAIL_RIGHT", "TRANSFORM_LONG_TAIL_LEFT"]: 
            groups["HEAVY_TAIL"].append(col)
        else: groups["NORMAL"].append(col)

    with open(groups_json_path, 'w', encoding='utf-8') as f: 
        json.dump(groups, f, indent=4, ensure_ascii=False)
        
    print(f"✅ Xong! Đã phân loại thành công: {groups_json_path}")

if __name__ == "__main__":
    run_profiler()