import os
import sys
import json
import time
import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
FINAL_EXP_NAME = "F_family_q99.5"
FINAL_STRATEGY_NAME = "Strategy_2_Aggressive"

TENSORS_ROOT = PROJECT_ROOT / "data_artifacts" / "gan_tensors"
FINAL_EXP_DIR = TENSORS_ROOT / FINAL_EXP_NAME
OUTPUT_DIR = PROJECT_ROOT / "final_report_outputs"

FINAL_WEIGHTS = {
    "lambda_adv": 15.0,
    "lambda_cov": 1.0,
    "lambda_l1": 0.1,
    "lambda_sat": 1.0,
}

DEFAULT_EPOCHS = 250
DEFAULT_BATCH_SIZE = 128

sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "Phase_3_GANTraining"))
sys.path.append(str(PROJECT_ROOT / "Phase_4_Evaluation"))

from Phase_3_GANTraining.main import train_single_experiment
from Phase_4_Evaluation.evaluator import evaluate_single_experiment


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_policy_summary():
    policy_path = FINAL_EXP_DIR / "adversarial_policy_S3.json"

    if not policy_path.exists():
        return {
            "policy_path": str(policy_path),
            "policy_exists": False,
            "error": "Missing adversarial_policy_S3.json",
        }

    with open(policy_path, "r", encoding="utf-8") as f:
        policy = json.load(f)

    zones = policy.get("zones", {})
    zone_summary = {}

    for zone_name, zone_data in zones.items():
        features = zone_data.get("features", [])
        zone_summary[zone_name] = {
            "allowed_variance": zone_data.get("allowed_variance"),
            "num_features": len(features),
            "features": features,
        }

    return {
        "policy_path": str(policy_path),
        "policy_exists": True,
        "metadata": policy.get("metadata", {}),
        "zones": zone_summary,
    }


def save_final_config(epochs, batch_size, eval_only):
    ensure_dirs()

    config = {
        "purpose": "Final locked experiment for report writing. No ablation. No auto-tuning.",
        "final_experiment": FINAL_EXP_NAME,
        "final_strategy": FINAL_STRATEGY_NAME,
        "final_weights": FINAL_WEIGHTS,
        "epochs": epochs,
        "batch_size": batch_size,
        "eval_only": eval_only,
        "policy_file": "adversarial_policy_S3.json",
        "experiment_dir": str(FINAL_EXP_DIR),
        "note": (
            "master_pipeline.py remains the exploratory auto-tuning/ablation script. "
            "This file only reproduces/evaluates the selected final configuration."
        ),
    }

    out_path = OUTPUT_DIR / "final_selected_config.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    return out_path


def run_final_experiment(epochs, batch_size, eval_only):
    if not FINAL_EXP_DIR.exists():
        raise FileNotFoundError(f"Missing final experiment folder: {FINAL_EXP_DIR}")

    print("=" * 80)
    print("FINAL SELECTED GAN BYPASS IDPS RUN")
    print("=" * 80)
    print(f"Experiment : {FINAL_EXP_NAME}")
    print(f"Strategy   : {FINAL_STRATEGY_NAME}")
    print(f"Weights    : {FINAL_WEIGHTS}")
    print(f"Epochs     : {epochs}")
    print(f"Batch size : {batch_size}")
    print(f"Eval only  : {eval_only}")
    print(f"Exp dir    : {FINAL_EXP_DIR}")
    print("=" * 80)

    if not eval_only:
        train_single_experiment(
            exp_dir=str(FINAL_EXP_DIR),
            epochs=epochs,
            batch_size=batch_size,
            lambda_adv=FINAL_WEIGHTS["lambda_adv"],
            lambda_cov=FINAL_WEIGHTS["lambda_cov"],
            lambda_l1=FINAL_WEIGHTS["lambda_l1"],
            lambda_sat=FINAL_WEIGHTS["lambda_sat"],
        )

    metrics = evaluate_single_experiment(str(FINAL_EXP_DIR))

    metrics["Final_Experiment"] = FINAL_EXP_NAME
    metrics["Final_Strategy"] = FINAL_STRATEGY_NAME
    metrics["lambda_adv"] = FINAL_WEIGHTS["lambda_adv"]
    metrics["lambda_cov"] = FINAL_WEIGHTS["lambda_cov"]
    metrics["lambda_l1"] = FINAL_WEIGHTS["lambda_l1"]
    metrics["lambda_sat"] = FINAL_WEIGHTS["lambda_sat"]
    metrics["epochs"] = epochs
    metrics["batch_size"] = batch_size
    metrics["eval_only"] = eval_only

    policy_summary = load_policy_summary()
    metrics["policy_exists"] = policy_summary.get("policy_exists", False)

    for zone_name, zone_info in policy_summary.get("zones", {}).items():
        metrics[f"{zone_name}_allowed_variance"] = zone_info.get("allowed_variance")
        metrics[f"{zone_name}_num_features"] = zone_info.get("num_features")

    return metrics, policy_summary


def save_outputs(metrics, policy_summary):
    ensure_dirs()

    df = pd.DataFrame([metrics])

    csv_path = OUTPUT_DIR / "final_selected_result.csv"
    md_path = OUTPUT_DIR / "final_selected_report_summary.md"
    policy_path = OUTPUT_DIR / "final_selected_policy_summary.json"

    df.to_csv(csv_path, index=False)

    with open(policy_path, "w", encoding="utf-8") as f:
        json.dump(policy_summary, f, indent=4, ensure_ascii=False)

    selected_cols = [
        "Final_Experiment",
        "Final_Strategy",
        "Baseline_Detect_%",
        "Evasion_Rate_%",
        "IDS_Confidence_%",
        "Mean_L1_Perturbation",
        "Max_L_inf_Drift",
        "Policy_Compliance_%",
        "lambda_adv",
        "lambda_cov",
        "lambda_l1",
        "lambda_sat",
        "epochs",
        "batch_size",
    ]

    existing_cols = [c for c in selected_cols if c in df.columns]
    report_df = df[existing_cols].copy()

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Final Selected Experiment Report Summary\n\n")

        f.write("## Final locked configuration\n\n")
        f.write(f"- Final experiment: `{FINAL_EXP_NAME}`\n")
        f.write(f"- Final strategy: `{FINAL_STRATEGY_NAME}`\n")
        f.write(f"- Loss weights: `{FINAL_WEIGHTS}`\n")
        f.write("- Policy file: `adversarial_policy_S3.json`\n")
        f.write("- Purpose: report-ready final run, no ablation, no auto-tuning.\n\n")

        f.write("## Result\n\n")
        f.write(report_df.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Reporting note\n\n")
        f.write(
            "This is the locked final selected configuration. "
            "Ablation results should be reported separately as supporting experiments, "
            "not mixed into the final reproduction script.\n"
        )

    print("\n[OK] Saved:")
    print(f"  - {csv_path}")
    print(f"  - {md_path}")
    print(f"  - {policy_path}")

    return csv_path, md_path, policy_path


def print_result(metrics):
    print("\n" + "=" * 80)
    print("FINAL RESULT")
    print("=" * 80)

    important_keys = [
        "Final_Experiment",
        "Final_Strategy",
        "Baseline_Detect_%",
        "Evasion_Rate_%",
        "IDS_Confidence_%",
        "Mean_L1_Perturbation",
        "Max_L_inf_Drift",
        "Policy_Compliance_%",
    ]

    for key in important_keys:
        if key in metrics:
            print(f"{key}: {metrics[key]}")


def main():
    parser = argparse.ArgumentParser(
        description="Run/evaluate the final selected GAN bypass IDPS configuration only."
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only evaluate existing fake malware. Use this if training already produced tensor_fake_malware.pt.",
    )

    args = parser.parse_args()

    ensure_dirs()

    config_path = save_final_config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_only=args.eval_only,
    )

    print(f"[OK] Saved config: {config_path}")

    start = time.time()

    metrics, policy_summary = run_final_experiment(
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_only=args.eval_only,
    )

    save_outputs(metrics, policy_summary)
    print_result(metrics)

    elapsed = time.time() - start
    print(f"\n[OK] Done in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
