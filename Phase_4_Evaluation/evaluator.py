import os
import json
import torch
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

def calculate_evasion_rate(clf, fake_malware_numpy):
    """
    Tính Evasion Rate: Tỷ lệ mã độc fake bị phân loại NHẦM thành Benign (0).
    """
    preds = clf.predict(fake_malware_numpy)
    # Vì nhãn thật là Malware (1), nên nếu dự đoán ra 0 tức là Evasion thành công
    evaded = np.sum(preds == 0)
    total = len(preds)
    return evaded / total

def check_policy_compliance(real_np, fake_np, policy):
    """
    Kiểm tra xem GAN có tuân thủ 100% Policy hay không.
    (Đóng góp học thuật cốt lõi của bài báo)
    """
    diff = np.abs(fake_np - real_np)
    compliance_results = {}
    total_violations = 0

    for zone_name, zone_data in policy["zones"].items():
        allowed_var = zone_data["allowed_variance"]
        features_idx = zone_data["features"]
        
        if not features_idx:
            continue
            
        # Lấy độ lệch lớn nhất trong vùng này
        zone_diff = diff[:, features_idx]
        max_diff = np.max(zone_diff)
        mean_diff = np.mean(zone_diff)
        
        # Cho phép sai số dấu phẩy động (epsilon = 1e-4)
        epsilon = 1e-4
        is_compliant = max_diff <= (allowed_var + epsilon)
        
        violations = np.sum(zone_diff > (allowed_var + epsilon))
        total_violations += violations

        compliance_results[zone_name] = {
            "max_drift": max_diff,
            "mean_drift": mean_diff,
            "allowed": allowed_var,
            "compliant": bool(is_compliant)
        }
        
    overall_compliance = 100.0 if total_violations == 0 else (1.0 - (total_violations / (real_np.shape[0] * real_np.shape[1]))) * 100
    
    return compliance_results, overall_compliance

def evaluate_single_experiment(exp_dir):
    exp_name = os.path.basename(exp_dir)
    print(f"\n[{exp_name}] Đang phân tích pháp y...")

    # 1. Nạp dữ liệu
    real_mal_pt = torch.load(os.path.join(exp_dir, "tensor_malware.pt"))
    fake_mal_pt = torch.load(os.path.join(exp_dir, "tensor_fake_malware.pt"))
    benign_pt = torch.load(os.path.join(exp_dir, "tensor_benign.pt"))
    
    # Dùng glob để tìm file adversarial_policy (vì theo tree của bạn file có đuôi _S3)
    policy_files = list(Path(exp_dir).glob("adversarial_policy*.json"))
    if not policy_files:
        raise FileNotFoundError(f"Không tìm thấy file adversarial_policy*.json trong {exp_dir}")
        
    with open(policy_files[0], 'r') as f:
        policy = json.load(f)

    X_real_mal = real_mal_pt.numpy()
    X_fake_mal = fake_mal_pt.numpy()
    X_benign = benign_pt.numpy()

    # 2. Xây dựng lại Surrogate IDS (kẻ thù)
    y_real_mal = np.ones(X_real_mal.shape[0])
    y_benign = np.zeros(X_benign.shape[0])
    X_train_full = np.vstack((X_real_mal, X_benign))
    y_train_full = np.concatenate((y_real_mal, y_benign))

    X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)
    
    clf = xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, eval_metric='logloss', n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)
    
    # Baseline Detection Rate trên Malware THẬT
    mal_test = X_test[y_test == 1]
    baseline_detect = accuracy_score(np.ones(len(mal_test)), clf.predict(mal_test))

    # 3. Tính Evasion Rate
    evasion_rate = calculate_evasion_rate(clf, X_fake_mal)
    
    # [PHẦN MỚI THÊM VÀO]: Xem xác suất tự tin của IDS (Càng thấp tức là GAN càng hiệu quả)
    probs_fake = clf.predict_proba(X_fake_mal)[:, 1] 
    ids_confidence = np.mean(probs_fake)

    # 4. Perturbation Budget (L1 & L_inf Norms)
    diff = np.abs(X_fake_mal - X_real_mal)
    mean_l1 = np.mean(diff)
    max_linf = np.max(diff)
    top_feature_drift = np.max(np.mean(diff, axis=0))

    # 5. Policy Compliance & Semantic Preservation
    compliance_details, overall_compliance = check_policy_compliance(X_real_mal, X_fake_mal, policy)

    # === THAY THẾ PHẦN IN KẾT QUẢ ĐỂ IN CHI TIẾT TỪNG ZONE ===
    print(f"   -> Baseline Detection (Real): {baseline_detect*100:.2f}%")
    print(f"   -> Evasion Rate (Fake)      : {evasion_rate*100:.2f}%")
    print(f"   -> IDS Avg Confidence       : {ids_confidence*100:.2f}%")
    print(f"   -> Perturbation (L1 / L_inf): {mean_l1:.4f} / {max_linf:.4f}")
    
    # [NÂNG CẤP MỚI]: IN CHI TIẾT TỪNG ZONE ĐỂ LÀM BẰNG CHỨNG
    print(f"   -> Tổng Policy Compliance   : {overall_compliance:.2f}%")
    print(f"      [Chi tiết các Zone]:")
    for zone, details in compliance_details.items():
        if details["compliant"]:
            status = "✅ An toàn"
        else:
            status = f"❌ VI PHẠM (Lệch thực tế: {details['max_drift']:.4f} > Ngưỡng cho phép: {details['allowed']:.4f})"
        
        # Cảnh báo gắt nếu chạm vào Critical
        if zone == "CRITICAL" and not details["compliant"]:
            status += " ⚠️ MÃ ĐỘC ĐÃ BỊ HỎNG CẤU TRÚC!"
            
        print(f"        - {zone}: {status}")

    return {
        "Experiment": exp_name,
        "Baseline_Detect_%": baseline_detect * 100,
        "Evasion_Rate_%": evasion_rate * 100,
        "IDS_Confidence_%": ids_confidence * 100,
        "Mean_L1_Perturbation": mean_l1,
        "Max_L_inf_Drift": max_linf,
        "Top_Feature_Drift": top_feature_drift,
        "Policy_Compliance_%": overall_compliance,
        "Critical_Max_Drift": compliance_details.get("CRITICAL", {}).get("max_drift", 0)
    }

def run_evaluation_pipeline():
    
    # Dùng path relative để trỏ về gốc của project (gan_bypass_idps)
    project_root = Path(__file__).resolve().parent.parent
    tensors_root = project_root / "data_artifacts" / "gan_tensors"
    
    # Bây giờ tensors_root đã là một Path object, hàm iterdir() sẽ hoạt động tốt
    exp_folders = sorted([f for f in tensors_root.iterdir() if f.is_dir()])
    print(f"[*] Bắt đầu Phase 4: Đánh giá Chất lượng Sinh Adversarial Malware trên {len(exp_folders)} Experiments...")

    results = []
    for folder in exp_folders:
        try:
            metrics = evaluate_single_experiment(str(folder))
            results.append(metrics)
        except Exception as e:
            print(f"   [!] Lỗi khi đánh giá {folder.name}: {e}")

    # Xuất báo cáo tổng hợp
    df = pd.DataFrame(results)
    print("\n" + "="*105)
    print(f"{'BẢNG TỔNG HỢP ABLATION STUDY (EVALUATION METRICS)':^105}")
    print("="*105)
    
    # Format hiển thị cho đẹp
    df_display = df.copy()
    for col in ["Baseline_Detect_%", "Evasion_Rate_%", "IDS_Confidence_%", "Policy_Compliance_%"]:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.2f}%")
    for col in ["Mean_L1_Perturbation", "Max_L_inf_Drift", "Top_Feature_Drift", "Critical_Max_Drift"]:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}")
        
    print(df_display.to_string(index=False))
    
    # Lưu file ra thư mục root của project
    output_csv = project_root / "ablation_evaluation_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Đã xuất báo cáo ra file: {output_csv.name}")

if __name__ == "__main__":
    run_evaluation_pipeline()