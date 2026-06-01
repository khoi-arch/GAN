import argparse
import os
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def load_tensor(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    return torch.load(path, map_location="cpu").detach().cpu().numpy()


def build_policy_xgb():
    """
    Y hệt model trong Phase_2_TargetMapping/04_space_mapper.py
    dùng để lập adversarial policy.
    """
    return xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )


def build_eval_xgb():
    """
    Y hệt model trong Phase_4_Evaluation/evaluator.py
    dùng để eval fake malware.
    """
    return xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )


def print_binary_metrics(y_true, y_pred, title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    print(f"Accuracy : {accuracy_score(y_true, y_pred):.6f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.6f}")
    print(f"Recall   : {recall_score(y_true, y_pred, zero_division=0):.6f}")
    print(f"F1-score : {f1_score(y_true, y_pred, zero_division=0):.6f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Benign", "Malware"],
            digits=6,
            zero_division=0,
        )
    )


def run_policy_mode(exp_dir):
    """
    Chạy giống Phase 2:
    - load tensor_malware.pt + tensor_benign.pt
    - label malware = 1, benign = 0
    - train_test_split có stratify=y
    - train XGBoost policy surrogate
    - in precision/recall/F1
    """
    malware_path = os.path.join(exp_dir, "tensor_malware.pt")
    benign_path = os.path.join(exp_dir, "tensor_benign.pt")

    X_malware = load_tensor(malware_path)
    X_benign = load_tensor(benign_path)

    y_malware = np.ones(X_malware.shape[0], dtype=np.int64)
    y_benign = np.zeros(X_benign.shape[0], dtype=np.int64)

    X = np.vstack((X_malware, X_benign))
    y = np.concatenate((y_malware, y_benign))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    clf = build_policy_xgb()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(f"\n[POLICY MODE] Experiment dir: {exp_dir}")
    print(f"X_malware: {X_malware.shape}")
    print(f"X_benign : {X_benign.shape}")
    print(f"Train    : {X_train.shape}")
    print(f"Test     : {X_test.shape}")

    print_binary_metrics(
        y_test,
        y_pred,
        "Surrogate IDS metrics — giống Phase 2 policy model",
    )


def run_eval_mode(exp_dir):
    """
    Chạy giống Phase 4:
    - load tensor_malware.pt + tensor_benign.pt
    - train_test_split không stratify
    - train lại XGBoost evaluator surrogate
    - in precision/recall/F1 trên real malware + benign test split
    - nếu có tensor_fake_malware.pt thì in thêm evasion rate fake malware
    """
    malware_path = os.path.join(exp_dir, "tensor_malware.pt")
    benign_path = os.path.join(exp_dir, "tensor_benign.pt")
    fake_path = os.path.join(exp_dir, "tensor_fake_malware.pt")

    X_real_mal = load_tensor(malware_path)
    X_benign = load_tensor(benign_path)

    y_real_mal = np.ones(X_real_mal.shape[0], dtype=np.int64)
    y_benign = np.zeros(X_benign.shape[0], dtype=np.int64)

    X_train_full = np.vstack((X_real_mal, X_benign))
    y_train_full = np.concatenate((y_real_mal, y_benign))

    X_train, X_test, y_train, y_test = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=42,
    )

    clf = build_eval_xgb()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(f"\n[EVAL MODE] Experiment dir: {exp_dir}")
    print(f"X_real_mal: {X_real_mal.shape}")
    print(f"X_benign  : {X_benign.shape}")
    print(f"Train     : {X_train.shape}")
    print(f"Test      : {X_test.shape}")

    print_binary_metrics(
        y_test,
        y_pred,
        "Surrogate IDS metrics — giống Phase 4 evaluator model",
    )

    # Baseline Detection Rate giống evaluator.py
    mal_test = X_test[y_test == 1]
    baseline_detect = accuracy_score(
        np.ones(len(mal_test), dtype=np.int64),
        clf.predict(mal_test),
    )
    print(f"\nBaseline Detection trên Real Malware test: {baseline_detect * 100:.4f}%")

    # Nếu đã train GAN và có fake malware thì đo thêm evasion
    if os.path.exists(fake_path):
        X_fake_mal = load_tensor(fake_path)
        fake_pred = clf.predict(X_fake_mal)

        # fake malware label thật về mặt bản chất vẫn là malware = 1
        y_fake_true = np.ones(X_fake_mal.shape[0], dtype=np.int64)

        fake_precision = precision_score(y_fake_true, fake_pred, zero_division=0)
        fake_recall = recall_score(y_fake_true, fake_pred, zero_division=0)
        fake_f1 = f1_score(y_fake_true, fake_pred, zero_division=0)

        evasion_rate = np.mean(fake_pred == 0)
        ids_confidence = np.mean(clf.predict_proba(X_fake_mal)[:, 1])

        print("\n" + "=" * 80)
        print("Fake malware evaluation — giống hướng Phase 4")
        print("=" * 80)
        print(f"Fake predicted as Benign / Evasion Rate: {evasion_rate * 100:.4f}%")
        print(f"IDS Avg Confidence Malware class       : {ids_confidence * 100:.4f}%")
        print(f"Fake Recall as Malware                 : {fake_recall:.6f}")
        print(f"Fake Precision as Malware              : {fake_precision:.6f}")
        print(f"Fake F1 as Malware                     : {fake_f1:.6f}")

        print("\nFake Malware Classification Report:")
        print(
            classification_report(
                y_fake_true,
                fake_pred,
                labels=[0, 1],
                target_names=["Benign", "Malware"],
                digits=6,
                zero_division=0,
            )
        )
    else:
        print("\nKhông thấy tensor_fake_malware.pt nên chỉ in metric real malware/benign.")


def main():
    parser = argparse.ArgumentParser(
        description="Check XGBoost surrogate IDS precision/recall/F1 on CPU."
    )
    parser.add_argument(
        "--exp-dir",
        required=True,
        help="Đường dẫn tới folder strategy, ví dụ data_artifacts/gan_tensors/F_family_q99.5",
    )
    parser.add_argument(
        "--mode",
        choices=["policy", "eval", "both"],
        default="both",
        help="policy = giống Phase 2, eval = giống Phase 4, both = chạy cả hai",
    )

    args = parser.parse_args()
    exp_dir = str(Path(args.exp_dir))

    if args.mode in ["policy", "both"]:
        run_policy_mode(exp_dir)

    if args.mode in ["eval", "both"]:
        run_eval_mode(exp_dir)


if __name__ == "__main__":
    main()