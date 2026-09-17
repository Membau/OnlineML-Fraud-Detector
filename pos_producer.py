
import argparse
import csv
import time
import uuid
import requests
import sys

parser = argparse.ArgumentParser(description="POS Transaction Stream Simulator")
parser.add_argument("--delay",  type=float, default=0.1,
                    help="Delay between transactions (seconds), default 0.1s")
parser.add_argument("--rows",   type=int,   default=1000,
                    help="Max transactions to send, default 1000")
parser.add_argument("--server", type=str,   default="http://localhost:8000",
                    help="URL of ML Engine server")
parser.add_argument("--feedback-delay", type=float, default=0.1,
                    help="Delay before sending actual label (Delayed Feedback), default 0.5s")
parser.add_argument("--data",   type=str,   default="data/creditcard.csv",
                    help="Path to CSV file")
args = parser.parse_args()

SERVER      = args.server.rstrip("/")
PREDICT_URL = f"{SERVER}/predict"
FEEDBACK_URL= f"{SERVER}/feedback"

def parse_row(row: dict) -> tuple[dict, int]:
    """Extract 'Class' label from features dict and parse as float."""
    label = int(float(row.pop("Class", 0)))
    row.pop("Time", None) # Remove Time column
    features = {k: float(v) for k, v in row.items() if v != ""}
    return features, label


def send_predict(features: dict, tx_id: str) -> dict:
    payload = {"features": features, "transaction_id": tx_id}
    resp = requests.post(PREDICT_URL, json=payload, timeout=5)
    resp.raise_for_status()
    return resp.json()


def send_feedback(features: dict, label: int, tx_id: str) -> dict:
    payload = {"features": features, "label": label, "transaction_id": tx_id}
    resp = requests.post(FEEDBACK_URL, json=payload, timeout=5)
    resp.raise_for_status()
    return resp.json()


def main():
    print(f"[POS Simulator] Connecting to {SERVER}")
    print(f"[POS Simulator] Data file: {args.data}")
    print(f"[POS Simulator] Delay: {args.delay}s | Feedback delay: {args.feedback_delay}s | Max rows: {args.rows}")
    print("-" * 60)

    try:
        health = requests.get(SERVER, timeout=3)
        print(f"[POS Simulator] Server status: {health.json().get('status', 'unknown')}")
    except Exception:
        print("[POS Simulator] Cannot connect to server. Make sure server.py is running!")
        sys.exit(1)

    count = 0
    fraud_count = 0
    blocked_count = 0

    try:
        with open(args.data, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)

            for raw_row in reader:
                if count >= args.rows:
                    break


                row_copy = dict(raw_row)
                features, true_label = parse_row(raw_row)
                tx_id = str(uuid.uuid4())[:8]
                try:
                    result = send_predict(features, tx_id)
                    prob   = result.get("fraud_probability", 0.0)
                    action = result.get("action", "?")

                    flag = " FRAUD" if true_label == 1 else "  normal"
                    blocked = " BLOCKED" if action == "BLOCK" else " ALLOWED"
                    print(f"[TX {tx_id}] {flag} | Prob={prob:.4f} | {blocked}")

                    if action == "BLOCK":
                        blocked_count += 1
                    if true_label == 1:
                        fraud_count += 1

                except Exception as e:
                    print(f"[TX {tx_id}] Predict error: {e}")
                    count += 1
                    continue
                time.sleep(args.feedback_delay)
                try:
                    fb_features, _ = parse_row(dict(row_copy))
                    fb_result = send_feedback(fb_features, true_label, tx_id)
                    roc_auc = fb_result.get("roc_auc", 0.0)
                    print(f"          ↳ Feedback sent | Label={true_label} | ROC-AUC={roc_auc:.4f}")
                except Exception as e:
                    print(f"          ↳ Feedback error: {e}")

                count += 1
                time.sleep(args.delay)

    except FileNotFoundError:
        print(f"[POS Simulator] File not found: {args.data}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[POS Simulator] Stopped by user.")

    print("-" * 60)
    print(f"[POS Simulator] Completed: {count} transactions | {fraud_count} frauds | {blocked_count} blocked")


if __name__ == "__main__":
    main()
