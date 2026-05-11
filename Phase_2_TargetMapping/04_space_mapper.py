import os
import json
import torch
import numpy as np
import xgboost as xgb
import shap
import warnings
import random
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

def run_space_mapper(data_dir="../data_artifacts"):
    # Kích hoạt tính nhất quán toàn cục
    set_global_seed(42)
    
    print("[1/5] Đang load Tensors và chuẩn bị dữ liệu Surrogate...")
    
    # Load Tensors
    malware_tensor = torch.load(os.path.join(data_dir, "tensor_malware.pt"))
    benign_tensor = torch.load(os.path.join(data_dir, "tensor_benign.pt"))
    
    X_malware = malware_tensor.numpy()
    X_benign = benign_tensor.numpy()
    
    y_malware = np.ones(X_malware.shape[0])
    y_benign = np.zeros(X_benign.shape[0])
    
    X = np.vstack((X_malware, X_benign))
    y = np.concatenate((y_malware, y_benign))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("[2/5] Đang train Surrogate IDS (XGBoost)...")
    clf = xgb.XGBClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.1,
        use_label_encoder=False, eval_metric='logloss', n_jobs=-1,
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Độ chính xác Surrogate: {acc*100:.2f}%")

    print("[3/5] Đang phân tích SHAP và Sensitivity...")
    
    # Latent Stratified Sampling
    print("      -> Gom cụm Latent Families để chống bias khi lấy mẫu...")
    
    # [NÂNG CẤP 2] Sửa logic số cụm để tránh KMeans sập khi dataset nhỏ
    n_latent_families = max(2, min(10, len(X_malware) // 50))
    kmeans_latent = KMeans(n_clusters=n_latent_families, random_state=42, n_init=10).fit(X_malware)
    latent_labels = kmeans_latent.labels_
    
    sample_size = min(2000, len(X_malware))
    
    if len(X_malware) > sample_size:
        try:
            # Cố gắng chia phân tầng (Stratified)
            X_sample, _, _, _ = train_test_split(
                X_malware, latent_labels, 
                train_size=sample_size, 
                stratify=latent_labels, 
                random_state=42
            )
        except ValueError:
            # Safeguard: Bắt lỗi Singleton Cluster -> Chuyển sang lấy mẫu ngẫu nhiên (Uniform Random)
            print("      [!] Cảnh báo: Stratified split thất bại. Chuyển sang lấy mẫu ngẫu nhiên (Uniform Sampling)...")
            sample_idx = np.random.choice(len(X_malware), sample_size, replace=False)
            X_sample = X_malware[sample_idx]
    else:
        X_sample = X_malware
    
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
    
    print("[4/5] Đang tính toán Risk Score và phân vùng Natural Breaks (1D K-Means)...")
    
    # Dùng Geometric Mean (Căn bậc 2 của tích) thay vì nhân trực tiếp
    risk_scores = np.sqrt(norm_shap * norm_sens)
    
    try:
        feature_names = get_feature_names(os.path.join(data_dir, "feature_groups.json"))
        # Sanity Check
        if len(feature_names) != X_malware.shape[1]:
            print(f"      [!] LỖI NGHIÊM TRỌNG: Số lượng feature name ({len(feature_names)}) không khớp với số cột Tensor ({X_malware.shape[1]})!")
            feature_names = [f"F_{i}" for i in range(X_malware.shape[1])]
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
        
    # Sort Metrics tổng từ Risk cao xuống thấp để view dễ nhìn
    feature_metrics.sort(key=lambda x: x["risk_score"], reverse=True)
    for z in zones_temp:
        zones_temp[z].sort(key=lambda x: x["risk_score"], reverse=True)
    
    print("[5/5] Đang xuất Adversarial Policy...")
    
    policy = {
        "metadata": {
            "surrogate_accuracy": float(acc),
            "total_features": len(feature_names),
            "sampling_size": sample_size,
            "clustering_method": "1D K-Means (n_init=50, Natural Breaks)",
            "risk_fusion_method": "Geometric Mean",
            "normalization": "Robust Scaling (P95 Clip)"
        },
        "zones": {
            "CRITICAL": {
                "description": "High Risk (Semantic Identity). Allowed variance: 2%",
                "allowed_variance": 0.02,
                "features": [f["index"] for f in zones_temp["CRITICAL"]]
            },
            "MEDIUM": {
                "description": "Moderate Risk. Allowed variance: 10%",
                "allowed_variance": 0.10,
                "features": [f["index"] for f in zones_temp["MEDIUM"]]
            },
            "FREE": {
                "description": "Low Risk (Blind spots). Allowed variance: 20%",
                "allowed_variance": 0.20,
                "features": [f["index"] for f in zones_temp["FREE"]]
            }
        },
        "detailed_metrics": feature_metrics
    }

    out_path = os.path.join(data_dir, "adversarial_policy.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(policy, f, indent=4)
        
    print(f"✅ Xong! Policy đã sẵn sàng tại: {out_path}")
    print(f"   -> Đã tự động chốt {len(zones_temp['CRITICAL'])} cột vào vùng CRITICAL.")
    print(f"   -> Đã tự động chốt {len(zones_temp['MEDIUM'])} cột vào vùng MEDIUM.")
    print(f"   -> Đã tự động chốt {len(zones_temp['FREE'])} cột vào vùng FREE.")

if __name__ == "__main__":
    run_space_mapper()