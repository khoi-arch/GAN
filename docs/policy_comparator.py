import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr

# Tắt bớt mấy cái warning vô duyên của Matplotlib
import warnings
warnings.filterwarnings('ignore')

def calculate_jaccard(list1, list2):
    """Tính độ tương đồng Jaccard (Tỷ lệ khớp giữa 2 tập hợp)"""
    set1, set2 = set(list1), set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def get_top_k_features(detailed_metrics, k=20):
    """Trích xuất danh sách tên của Top K features nguy hiểm nhất"""
    sorted_metrics = sorted(detailed_metrics, key=lambda x: x["risk_score"], reverse=True)
    return [m["name"] for m in sorted_metrics[:k]]

def load_and_compare():
    # 1. SETUP ĐƯỜNG DẪN (Tự động nhận diện folder gan_tensors)
    script_dir = Path(__file__).resolve().parent
    tensors_root = script_dir / "data_artifacts" / "gan_tensors"
    
    if not tensors_root.exists():
        print(f"❌ KHÔNG TÌM THẤY DỮ LIỆU TẠI: {tensors_root}")
        return

    settings = ["S1", "S2", "S3", "S4", "S5"]
    exp_folders = sorted([f.name for f in tensors_root.iterdir() if f.is_dir()])
    
    results = []
    top_k_value = 20 # Chốt Top 20 theo ý ní

    print(f"[*] Đang xử lý 35 kịch bản (7 Folders x 5 Settings)...")

    for exp in exp_folders:
        policies = {}
        for s in settings:
            p_path = tensors_root / exp / f"adversarial_policy_{s}.json"
            if p_path.exists():
                with open(p_path, 'r', encoding='utf-8') as f:
                    policies[s] = json.load(f)
        
        # Ground Truth là bản FULL (S5)
        if "S5" not in policies:
            continue

        full_policy = policies["S5"]
        # Lấy tên feature và score của bản FULL để làm mốc so sánh Rank
        full_metrics_map = {m["name"]: m["risk_score"] for m in full_policy["detailed_metrics"]}
        sorted_names = sorted(full_metrics_map.keys())
        full_vector = [full_metrics_map[name] for name in sorted_names]
        
        # Lấy danh sách Top 20 của bản FULL
        full_top20 = get_top_k_features(full_policy["detailed_metrics"], k=top_k_value)

        for s in settings:
            p = policies.get(s)
            if not p: continue

            # Lấy số lượng samples thật (500, 1000, 2000...) làm trục X
            samples = p["metadata"]["samples_used"]

            # --- METRIC 1: RANK CORRELATION (SPEARMAN RHO) ---
            curr_metrics_map = {m["name"]: m["risk_score"] for m in p["detailed_metrics"]}
            curr_vector = [curr_metrics_map.get(name, 0.0) for name in sorted_names]
            rho, _ = spearmanr(full_vector, curr_vector)

            # --- METRIC 2: TOP-20 OVERLAP (JACCARD) ---
            curr_top20 = get_top_k_features(p["detailed_metrics"], k=top_k_value)
            overlap = calculate_jaccard(curr_top20, full_top20)

            # --- METRIC 3: RISK MAE (SAI SỐ TRUNG BÌNH) ---
            mae = np.mean([abs(full_metrics_map[n] - curr_metrics_map.get(n, 0)) for n in sorted_names])

            results.append({
                "Experiment": exp,
                "Setting": s,
                "Samples": samples,
                "Rank_Correlation": rho,
                "Top20_Overlap": overlap,
                "Risk_MAE": mae
            })

    # Chuyển về DataFrame để Seaborn quẩy
    df = pd.DataFrame(results)
    
    # 2. XUẤT BẢNG SUMMARY (Tính Mean qua 7 Experiments)
    print("\n" + "="*95)
    print(f"{'STATISTICAL STABILITY ANALYSIS (MEAN ± SD)':^95}")
    print("="*95)
    summary = df.groupby("Setting")[["Samples", "Rank_Correlation", "Top20_Overlap", "Risk_MAE"]].agg(['mean', 'std'])
    print(summary)
    
    # Lưu CSV chi tiết
    df.to_csv("ablation_study_final.csv", index=False)

    # 3. VẼ BIỂU ĐỒ (CÓ VÙNG MỜ SD)
    sns.set(style="whitegrid", palette="muted")
    plt.figure(figsize=(12, 6))

    # Plot 1: Overlap & Rank Correlation (Trục Y bên trái)
    ax = plt.gca()
    
    # Rank Correlation với vùng mờ SD
    sns.lineplot(data=df, x="Samples", y="Rank_Correlation", marker='o', 
                 label="Rank Correlation (Spearman Rho)", errorbar="sd", ax=ax, linewidth=2.5)
    
    # Top-20 Overlap với vùng mờ SD
    sns.lineplot(data=df, x="Samples", y="Top20_Overlap", marker='s', 
                 label=f"Top-{top_k_value} Overlap (Jaccard)", errorbar="sd", ax=ax, linewidth=2.5)

    plt.xscale('log') # Reviewer cực thích log-scale để soi điểm gãy
    plt.title(f"Malware Feature Stability: Convergence Analysis (n=7 Experiments)", fontsize=15, fontweight='bold')
    plt.xlabel("Malware Samples (Log Scale)", fontsize=12)
    plt.ylabel("Stability Score (0.0 - 1.0)", fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(loc='lower right', frameon=True)
    
    # Thêm chú thích cho Reviewer
    plt.text(df['Samples'].min(), 0.05, "* Shadow areas represent Standard Deviation (SD) across 7 preprocessing methods.", 
             fontsize=9, style='italic', color='gray')

    plt.tight_layout()
    plt.savefig("scientific_stability_convergence.png", dpi=300)
    print(f"\n✅ Đồ thị 'uy tín' đã sẵn sàng: scientific_stability_convergence.png")
    plt.show()

if __name__ == "__main__":
    load_and_compare()