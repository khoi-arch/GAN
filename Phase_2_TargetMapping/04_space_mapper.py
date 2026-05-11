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
    
    # [4/5] Đang tính toán Risk Score và phân vùng Natural Breaks (1D K-Means)...
    # Dùng Geometric Mean (Căn bậc 2 của tích) thay vì nhân trực tiếp
    risk_scores = np.sqrt(norm_shap * norm_sens)
    
    try:
        feature_names = get_feature_names(groups_path)
    except Exception:
        feature_names = [f"F_{i}" for i in range(X_malware.shape[1])]

    # Dùng K-Means 1D để tự động tìm ngưỡng cắt (Natural Breaks) với n_init=50 để ổn định
    risk_matrix = risk_scores.reshape(-1, 1)
    kmeans_1d = KMeans(n_clusters=3, random_state=42, n_init=50).fit(risk_matrix)
    centers = kmeans_1d.cluster_centers_.flatten()
    
    # Sắp xếp các cụm: 0 = Nhỏ nhất (FREE), 1 = Trung bình (MEDIUM), 2 = Lớn nhất (CRITICAL)
    sorted_centers_idx = np.argsort(centers)
    label_free = sorted_centers_idx[0]
    label_medium = sorted_centers_idx[1]
    label_critical = sorted_centers_idx[2]
    
    feature_labels = kmeans_1d.labels_
    
    # Nhóm features vào các dictionary tạm
    zones_temp = {"CRITICAL": [], "MEDIUM": [], "FREE": []}
    feature_metrics = []

    for i in range(len(feature_names)):
        f_label = feature_labels[i]
        
        if f_label == label_critical: zone_name = "CRITICAL"
        elif f_label == label_medium: zone_name = "MEDIUM"
        else: zone_name = "FREE"
        
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
        
    # Sort Metrics
    feature_metrics.sort(key=lambda x: x["risk_score"], reverse=True)
    
    # [5/5] Đang xuất Adversarial Policy...
    policy = {
        "metadata": {
            "experiment": os.path.basename(data_dir),
            "setting": s_name,
            "samples_used": actual_size,
            "surrogate_accuracy": float(acc),
            "total_features": len(feature_names),
            "clustering_method": "1D K-Means (n_init=50, Natural Breaks)",
            "risk_fusion_method": "Geometric Mean",
            "normalization": "Robust Scaling (P95 Clip)"
        },
        "zones": {
            "CRITICAL": {"allowed_variance": 0.02, "features": [f["index"] for f in zones_temp["CRITICAL"]]},
            "MEDIUM": {"allowed_variance": 0.10, "features": [f["index"] for f in zones_temp["MEDIUM"]]},
            "FREE": {"allowed_variance": 0.20, "features": [f["index"] for f in zones_temp["FREE"]]}
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