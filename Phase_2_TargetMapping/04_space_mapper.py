import os
import json
import torch
import numpy as np
import xgboost as xgb
import shap
import warnings
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

# [NÂNG CẤP 1] Ép Global Seed để đảm bảo Reproducibility (Tái tạo kết quả 100%)
def set_global_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_feature_names(groups_path):
    """
    Đọc file groups để lấy lại đúng danh sách và thứ tự các cột đã đưa vào Tensor.
    [NÂNG CẤP 3] Lọc bỏ chính xác meta_cols y hệt như Phase 1 để đảm bảo Index khớp 100% khi Sorted.
    """
    with open(groups_path, 'r', encoding='utf-8') as f:
        groups = json.load(f)
    
    meta_cols = ['Class', 'Category', 'Binary_Label', 'label_L1', 'label_L2', 'label_L3']
    valid_cols = []
    
    for group_name, cols in groups.items():
        if group_name != "DROP":
            for c in cols:
                if c not in meta_cols and c not in valid_cols:
                    valid_cols.append(c)
                    
    # Bắt buộc phải sort vì Phase 1 (03_tensor_builder.py) đã sort khi build Tensor
    return sorted(valid_cols)

def calculate_sensitivity(clf, X, eps_ratio=0.05, k_trials=5):
    """
    Sensitivity Probing:
    Đo lường độ nhạy của mô hình bằng cách bơm nhiễu Gaussian, lấy P95 và trung bình qua k_trials.
    """
    print(f"      -> Đang chạy Sensitivity Probing ({k_trials} trials, P95 Random Signed Perturbation)...")
    base_probs = clf.predict_proba(X)[:, 1] # Xác suất dự đoán là Malware
    n_samples, n_features = X.shape
    
    stds = np.std(X, axis=0) + 1e-9 # Tránh chia 0
    accumulated_sensitivities = np.zeros(n_features)
    
    # Monte Carlo Approximation: Chạy k_trials lần để ổn định variance
    for trial in range(k_trials):
        trial_sensitivities = np.zeros(n_features)
        for i in range(n_features):
            X_perturbed = X.copy()
            
            # Bơm nhiễu Gaussian (Random Signed) thay vì tịnh tiến +eps một chiều
            noise = np.random.normal(loc=0.0, scale=eps_ratio * stds[i], size=n_samples)
            X_perturbed[:, i] += noise
            
            new_probs = clf.predict_proba(X_perturbed)[:, 1]
            delta = np.abs(new_probs - base_probs)
            
            # Dùng Percentile 95 (P95) để bắt 'worst-case' thay vì Mean
            trial_sensitivities[i] = np.percentile(delta, 95)
            
        accumulated_sensitivities += trial_sensitivities
        
    return accumulated_sensitivities / k_trials

def process_logic_original(data_dir, groups_path, s_name, s_size):
    """
    HÀM BÊ NGUYÊN 100% LOGIC CŨ CỦA NÍ VÀO ĐÂY ĐỂ CHẠY THEO SETTING S1-S5

    [PATCH POLICY 2026]
    - Giữ nguyên logic load tensor, train XGBoost, SHAP, sensitivity và latent stratified sampling.
    - Sửa phần risk fusion + chia zone policy:
        + Không dùng Geometric Mean vì sqrt(SHAP * Sensitivity) dễ collapse về 0
          nếu sensitivity_norm = 0, dù SHAP vẫn cao.
        + Dùng Weighted Sum: 0.7 * SHAP + 0.3 * Sensitivity.
        + Không dùng KMeans 1D để tránh CRITICAL/MEDIUM rỗng.
        + Dùng rank-based zoning:
            Top 10% risk  -> CRITICAL
            Next 20% risk -> MEDIUM
            Remaining     -> FREE
        + Đồng bộ allowed_variance:
            CRITICAL = 0.02
            MEDIUM   = 0.10
            FREE     = 0.20
    """
    print(f"\n[RUN] Experiment: {os.path.basename(data_dir)} | Setting: {s_name} (Size: {s_size if s_size else 'FULL'})")

    # [1/5] Đang load Tensors và chuẩn bị dữ liệu Surrogate...
    malware_tensor = torch.load(os.path.join(data_dir, "tensor_malware.pt"))
    benign_tensor = torch.load(os.path.join(data_dir, "tensor_benign.pt"))
    
    X_malware = malware_tensor.numpy()
    X_benign = benign_tensor.numpy()
    
    y_malware = np.ones(X_malware.shape[0])
    y_benign = np.zeros(X_benign.shape[0])
    
    X = np.vstack((X_malware, X_benign))
    y = np.concatenate((y_malware, y_benign))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # [2/5] Đang train Surrogate IDS (XGBoost)...
    clf = xgb.XGBClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.1,
        use_label_encoder=False, eval_metric='logloss', n_jobs=-1,
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Độ chính xác Surrogate: {acc*100:.2f}%")

    # [3/5] Đang phân tích SHAP và Sensitivity...
    # Latent Stratified Sampling logic [NÂNG CẤP 2]
    n_latent_families = max(2, min(10, len(X_malware) // 50))
    kmeans_latent = KMeans(n_clusters=n_latent_families, random_state=42, n_init=10).fit(X_malware)
    latent_labels = kmeans_latent.labels_
    
    # ĐIỀU CHỈNH SIZE THEO THÍ NGHIỆM S1-S5 CỦA NÍ
    if s_size is None or s_size >= len(X_malware):
        actual_size = len(X_malware)
        X_sample = X_malware
    else:
        actual_size = s_size
        try:
            X_sample, _, _, _ = train_test_split(
                X_malware, latent_labels, 
                train_size=actual_size, 
                stratify=latent_labels, 
                random_state=42
            )
        except ValueError:
            sample_idx = np.random.choice(len(X_malware), actual_size, replace=False)
            X_sample = X_malware[sample_idx]
    
    # Tính SHAP
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_sample)
    raw_shap_importance = np.mean(np.abs(shap_values), axis=0)
    
    # Robust Scaling cho SHAP để chống Outlier Domination
    p95_shap = np.percentile(raw_shap_importance, 95) + 1e-9
    norm_shap = np.clip(raw_shap_importance / p95_shap, 0.0, 1.0)
    
    # Tính SENSITIVITY
    raw_sensitivities = calculate_sensitivity(clf, X_sample, eps_ratio=0.05, k_trials=5)
    
    # Áp dụng Robust Scaling tương tự cho Sensitivity để đồng bộ không gian
    p95_sens = np.percentile(raw_sensitivities, 95) + 1e-9
    norm_sens = np.clip(raw_sensitivities / p95_sens, 0.0, 1.0)
    
    # [4/5] Đang tính toán Risk Score và phân vùng Rank-based Policy...
    #
    # LÝ DO SỬA:
    # Logic cũ dùng Geometric Mean:
    #     risk_scores = sqrt(norm_shap * norm_sens)
    # Nếu norm_sens = 0 thì risk_score = 0, dù norm_shap cao.
    # Điều này làm policy dễ collapse thành all-FREE.
    #
    # Logic mới:
    #     risk_scores = 0.7 * norm_shap + 0.3 * norm_sens
    # để giữ tín hiệu SHAP quan trọng, đồng thời vẫn tính sensitivity.
    risk_alpha = 0.70
    risk_scores = risk_alpha * norm_shap + (1.0 - risk_alpha) * norm_sens
    
    try:
        feature_names = get_feature_names(groups_path)
    except Exception:
        feature_names = [f"F_{i}" for i in range(X_malware.shape[1])]

    n_features = len(feature_names)

    if n_features != X_malware.shape[1]:
        print(f"      [!] Cảnh báo: số feature_names ({n_features}) khác tensor shape ({X_malware.shape[1]}).")
        print(f"      [!] Sẽ dùng số nhỏ hơn để tránh lệch index.")
        n_features = min(n_features, X_malware.shape[1])
        feature_names = feature_names[:n_features]
        risk_scores = risk_scores[:n_features]
        norm_shap = norm_shap[:n_features]
        norm_sens = norm_sens[:n_features]

    # Rank-based zoning để đảm bảo policy không bị all-FREE.
    # Với 52 feature:
    #   CRITICAL ~= 5 feature
    #   MEDIUM   ~= 10 feature
    #   FREE     ~= 37 feature
    critical_ratio = 0.10
    medium_ratio = 0.20

    n_critical = max(1, int(round(n_features * critical_ratio)))
    n_medium = max(1, int(round(n_features * medium_ratio)))

    # Đảm bảo còn ít nhất 1 feature cho FREE nếu feature quá ít
    if n_critical + n_medium >= n_features:
        n_medium = max(0, n_features - n_critical - 1)

    sorted_idx = np.argsort(-risk_scores)

    critical_idx = set(sorted_idx[:n_critical].tolist())
    medium_idx = set(sorted_idx[n_critical:n_critical + n_medium].tolist())

    # Nhóm features vào các dictionary tạm
    zones_temp = {"CRITICAL": [], "MEDIUM": [], "FREE": []}
    feature_metrics = []

    for i in range(n_features):
        if i in critical_idx:
            zone_name = "CRITICAL"
        elif i in medium_idx:
            zone_name = "MEDIUM"
        else:
            zone_name = "FREE"
        
        metric = {
            "index": i,
            "name": feature_names[i],
            "risk_score": float(risk_scores[i]),
            "shap_norm": float(norm_shap[i]),
            "sensitivity_norm": float(norm_sens[i]),
            "zone": zone_name
        }
        feature_metrics.append(metric)
        zones_temp[zone_name].append(metric)
        
    # Sort Metrics theo risk giảm dần để dễ đọc trong JSON
    feature_metrics.sort(key=lambda x: x["risk_score"], reverse=True)

    print("      -> Policy zone counts:")
    print(f"         CRITICAL: {len(zones_temp['CRITICAL'])} features")
    print(f"         MEDIUM  : {len(zones_temp['MEDIUM'])} features")
    print(f"         FREE    : {len(zones_temp['FREE'])} features")
    
    # [5/5] Đang xuất Adversarial Policy...
    policy = {
        "metadata": {
            "experiment": os.path.basename(data_dir),
            "setting": s_name,
            "samples_used": actual_size,
            "surrogate_accuracy": float(acc),
            "total_features": len(feature_names),
            "zoning_method": "Rank-based zoning: Top 10% CRITICAL, Next 20% MEDIUM, Remaining FREE",
            "risk_fusion_method": "Weighted Sum: 0.70*SHAP + 0.30*Sensitivity",
            "normalization": "Robust Scaling (P95 Clip)",
            "zone_counts": {
                "CRITICAL": len(zones_temp["CRITICAL"]),
                "MEDIUM": len(zones_temp["MEDIUM"]),
                "FREE": len(zones_temp["FREE"])
            },
            "allowed_variance_policy": {
                "CRITICAL": 0.02,
                "MEDIUM": 0.10,
                "FREE": 0.20
            }
        },
        "zones": {
            "CRITICAL": {
                "allowed_variance": 0.02,
                "features": [f["index"] for f in zones_temp["CRITICAL"]]
            },
            "MEDIUM": {
                "allowed_variance": 0.10,
                "features": [f["index"] for f in zones_temp["MEDIUM"]]
            },
            "FREE": {
                "allowed_variance": 0.20,
                "features": [f["index"] for f in zones_temp["FREE"]]
            }
        },
        "detailed_metrics": feature_metrics
    }

    # Lưu file riêng biệt để ní dễ so sánh
    out_path = os.path.join(data_dir, f"adversarial_policy_{s_name}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(policy, f, indent=4)
    print(f"      ✅ Đã lưu Policy tại: {out_path}")

def run_auto_experiment():
    set_global_seed(42)
    
    # Map đường dẫn Tree
    script_dir = Path(__file__).resolve().parent
    artifacts_dir = script_dir.parent / "data_artifacts"
    tensors_root = artifacts_dir / "gan_tensors"
    groups_path = artifacts_dir / "feature_groups.json"
    
    # 5 Setting của ní để CHỨNG MINH
    settings = {
        "S1": 500,
        "S2": 1000,
        "S3": 2000,
        "S4": 5000,
        "S5": None # FULL (~20k)
    }

    exp_folders = sorted([f for f in tensors_root.iterdir() if f.is_dir()])
    
    print(f"[*] KHỞI ĐỘNG CHẾ ĐỘ AUTO-ABLATION (7 FOLDERS X 5 SETTINGS)")

    for folder in exp_folders:
        for s_name, s_size in settings.items():
            try:
                process_logic_original(str(folder), str(groups_path), s_name, s_size)
            except Exception as e:
                print(f"      [!] Lỗi tại {folder.name} ({s_name}): {e}")

if __name__ == "__main__":
    run_auto_experiment()