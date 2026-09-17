import argparse
import csv
import uuid
import requests
import sys
import time

parser = argparse.ArgumentParser(description="Turbo POS Transaction Simulator")
parser.add_argument("--rows", type=int, default=5000,
                    help="Max transactions to send, default 5000")
parser.add_argument("--server", type=str, default="http://localhost:8000",
                    help="URL of ML Engine server")
parser.add_argument("--data", type=str, default="data/creditcard.csv",
                    help="Path to CSV file")
args = parser.parse_args()

SERVER = args.server.rstrip("/")
PREDICT_URL = f"{SERVER}/predict"
FEEDBACK_URL = f"{SERVER}/feedback"

def parse_row(row: dict) -> tuple[dict, int]:
    label = int(float(row.pop("Class", 0)))
    row.pop("Time", None) # Remove Time column
    features = {k: float(v) for k, v in row.items() if v != ""}
    return features, label

def main():
    print(f"[BOOST MODE] Starting high-speed data stream to {SERVER}")
    print(f"[BOOST] Data: {args.data} | Rows: {args.rows}")
    print("-" * 60)

    # Use Session to keep HTTP connection alive (much faster than individual requests)
    session = requests.Session()

    try:
        health = session.get(SERVER, timeout=3)
        print(f"[BOOST] Server status: {health.json().get('status', 'unknown')}")
    except Exception:
        print("[BOOST] Could not connect to server. Please start the server first!")
        sys.exit(1)

    count = 0
    fraud_count = 0
    start_time = time.time()

    try:
        with open(args.data, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)

            for raw_row in reader:
                if count >= args.rows:
                    break

                features, true_label = parse_row(raw_row)
                tx_id = str(uuid.uuid4())[:8]

                # Send Inference (Predict)
                try:
                    pred_resp = session.post(PREDICT_URL, json={"features": features, "transaction_id": tx_id})
                except Exception as e:
                    print(f"Prediction send error: {e}")
                    continue

                # Send Feedback immediately (No Delay)
                try:
                    session.post(FEEDBACK_URL, json={"features": features, "label": true_label, "transaction_id": tx_id})
                except Exception as e:
                    print(f"Feedback send error: {e}")

                if true_label == 1:
                    fraud_count += 1
                
                count += 1

                # Only print log every 500 transactions to not slow down the terminal
                if count % 500 == 0:
                    print(f"[BOOST] Processed {count}/{args.rows} transactions...")

    except FileNotFoundError:
        print(f"[BOOST] File not found: {args.data}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[BOOST] Stopped by user.")

    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"[BOOST] Completed: {count} transactions ({fraud_count} frauds)")
    print(f"[BOOST] Time taken: {elapsed:.2f} seconds (Speed: {count/elapsed:.0f} tx/s)")
    print("[BOOST] Please open the Streamlit dashboard to view the final ROC-AUC score!")

if __name__ == "__main__":
    main()
