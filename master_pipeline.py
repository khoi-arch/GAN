import os
import sys
import time
import pandas as pd
from pathlib import Path
import traceback

# Lấy đường dẫn tuyệt đối của thư mục chứa file master_pipeline.py
PROJECT_ROOT = Path(__file__).resolve().parent

# Thêm thư mục gốc và các thư mục Phase vào PYTHONPATH
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "Phase_3_GANTraining"))
sys.path.append(str(PROJECT_ROOT / "Phase_4_Evaluation"))
# ==============================

# Bây giờ các lệnh import sẽ hoạt động bình thường
from Phase_3_GANTraining.main import train_single_experiment
from Phase_4_Evaluation.evaluator import evaluate_single_experiment

STRATEGIES = {
    "Strategy_1_Cautious":  {"adv": 1.0,  "cov": 5.0,  "l1": 2.0,  "sat": 10.0},
    "Strategy_2_Aggressive":{"adv": 10.0, "cov": 1.0,  "l1": 0.1,  "sat": 1.0},
    "Strategy_3_Manifold":  {"adv": 5.0,  "cov": 10.0, "l1": 0.5,  "sat": 2.0},
    "Strategy_4_Surgical":  {"adv": 8.0,  "cov": 2.0,  "l1": 5.0,  "sat": 1.0}
}

def phase1_auto_tuning(target_exp_dir):
    print("\n" + "="*80)
    print("🧠 BƯỚC 1: AUTO TUNING (MÀI KIẾM & XUẤT FILE BẰNG CHỨNG)")
    print("="*80)
    
    tuning_results = []
    
    for strategy_name, weights in STRATEGIES.items():
        print(f"\n[*] Đang thử nghiệm: {strategy_name}")
        try:
            train_single_experiment(
                exp_dir=str(target_exp_dir), 
                epochs=250, 
                batch_size=128,
                lambda_adv=weights['adv'],
                lambda_cov=weights['cov'],
                lambda_l1=weights['l1'],
                lambda_sat=weights['sat']
            )
            
            metrics = evaluate_single_experiment(str(target_exp_dir))
            metrics['Strategy'] = strategy_name
            metrics['weights'] = weights 
            tuning_results.append(metrics)
            time.sleep(2)
            
        except Exception as e:
            print(f"[!] Lỗi tại {strategy_name}: {e}")
            traceback.print_exc()
            
    df_tuning = pd.DataFrame(tuning_results)
    
    # [NÂNG CẤP MỚI]: LƯU FILE BẰNG CHỨNG CỦA PHASE 1
    evidence_path = PROJECT_ROOT / "phase1_tuning_evidence.csv"
    df_tuning.to_csv(evidence_path, index=False)
    print(f"\n💾 Đã lưu file bằng chứng Tuning tại: {evidence_path.name}")
    
    return df_tuning

def phase2_select_best_strategy(df_results):
    print("\n" + "="*80)
    print("🧠 BƯỚC 2: AI RA QUYẾT ĐỊNH (BỘ LỌC SỐNG CÒN)")
    print("="*80)
    
    for col in ['Evasion_Rate_%', 'Policy_Compliance_%']:
        if col in df_results.columns and df_results[col].dtype == object:
             df_results[col] = df_results[col].astype(str).str.replace('%', '').astype(float)
             
    valid_strategies = []
    print("   [Log Phân Tích Các Chiến Lược]:")
    
    # [NÂNG CẤP MỚI]: IN LOG CHI TIẾT LÝ DO CHỌN / LOẠI
    for index, row in df_results.iterrows():
        strat_name = row['Strategy']
        policy_score = row['Policy_Compliance_%']
        evasion_score = row['Evasion_Rate_%']
        
        if policy_score < 90.0:
            print(f"      ❌ LOẠI {strat_name}: Policy = {policy_score}% (< 90%). Mã độc bị hỏng cấu trúc!")
        else:
            print(f"      ✅ GIỮ {strat_name}: Policy = {policy_score}% | Evasion = {evasion_score}%")
            valid_strategies.append(row)
            
    df_valid = pd.DataFrame(valid_strategies)
    
    if df_valid.empty:
        print("\n⚠️ CẢNH BÁO: Tất cả chiến lược đều làm hỏng file (>10% lỗi). Trả về chiến lược Cautious (An toàn nhất).")
        return STRATEGIES["Strategy_1_Cautious"], "Strategy_1_Cautious"
        
    best_row = df_valid.sort_values(by='Evasion_Rate_%', ascending=False).iloc[0]
    
    print(f"\n🏆 ĐÃ CHỌN CHIẾN LƯỢC TỐT NHẤT: {best_row['Strategy']}")
    print(f"   -> Đây là chiến lược cân bằng tốt nhất giữa sức tấn công ({best_row['Evasion_Rate_%']}%) và độ toàn vẹn ({best_row['Policy_Compliance_%']}%)")
    
    return best_row['weights'], best_row['Strategy']

def phase3_full_ablation_run(tensors_root, best_weights, best_strategy_name):
    print("\n" + "="*80)
    print(f"🧠 BƯỚC 3: CÀN QUÉT 7 ABLATION TENSORS VỚI {best_strategy_name}")
    print("="*80)
    
    exp_folders = sorted([f for f in tensors_root.iterdir() if f.is_dir()])
    final_results = []
    
    for folder in exp_folders:
        try:
            train_single_experiment(
                exp_dir=str(folder), 
                epochs=250, 
                batch_size=128,
                lambda_adv=best_weights['adv'],
                lambda_cov=best_weights['cov'],
                lambda_l1=best_weights['l1'],
                lambda_sat=best_weights['sat']
            )
            metrics = evaluate_single_experiment(str(folder))
            final_results.append(metrics)
        except Exception as e:
            print(f"   [!] Lỗi khi xử lý {folder.name}: {e}")

    df_final = pd.DataFrame(final_results)
    output_path = PROJECT_ROOT / f"final_ablation_report_{best_strategy_name}.csv"
    df_final.to_csv(output_path, index=False)
    
    print("\n🎉 HOÀN TẤT! Báo cáo cuối cùng đã được lưu tại:", output_path.name)

if __name__ == "__main__":
    tensors_root = PROJECT_ROOT / "data_artifacts" / "gan_tensors"
    tuning_target_dir = tensors_root / "G_family_q99.9"
    
    df_tuning = phase1_auto_tuning(tuning_target_dir)
    best_weights, best_strategy_name = phase2_select_best_strategy(df_tuning)
    phase3_full_ablation_run(tensors_root, best_weights, best_strategy_name)