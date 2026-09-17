from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import threading
import time
import os
import pickle

from river import linear_model, preprocessing, metrics, compose, optim

app = FastAPI(title="Fraud Detection Streaming Engine", version="1.0.0")

MODEL_SGD_FILE = "model_sgd.pkl"
MODEL_ADAM_FILE = "model_adam.pkl"

if os.path.exists(MODEL_SGD_FILE) and os.path.exists(MODEL_ADAM_FILE):
    print("[Server] Successfully loaded pretrained models from .pkl files!")
    with open(MODEL_SGD_FILE, "rb") as f:
        model_sgd = pickle.load(f)
    with open(MODEL_ADAM_FILE, "rb") as f:
        model_adam = pickle.load(f)
else:
    print("[Server] .pkl files not found. Starting with empty models...")
    opt_sgd = optim.SGD(lr = 0.1)
    model_sgd = compose.Pipeline(
        preprocessing.StandardScaler(),
        linear_model.LogisticRegression(optimizer = opt_sgd)
    )

    opt_adam = optim.Adam(lr = 0.01)
    model_adam = compose.Pipeline(
        preprocessing.StandardScaler(),
        linear_model.LogisticRegression(optimizer = opt_adam)
    )

metric_sgd = metrics.ROCAUC()
metric_adam = metrics.ROCAUC()

acc_sgd = metrics.Accuracy()
acc_adam = metrics.Accuracy()

cm_sgd = metrics.ConfusionMatrix()
cm_adam = metrics.ConfusionMatrix()

transaction_log = []
LOG_MAX = 500
log_lock = threading.Lock()

stats = {
    "total": 0,
    "fraud_detected": 0,
    "blocked": 0,
    "roc_auc_sgd": 0.0,
    "roc_auc_adam": 0.0,
    "acc_sgd": 0.0,
    "acc_adam": 0.0,
    "cm_sgd": {"tp":0, "fp":0, "tn":0, "fn":0},
    "cm_adam": {"tp":0, "fp":0, "tn":0, "fn":0},
}


class Transaction(BaseModel):
    features: dict  
    transaction_id: Optional[str] = None


class FeedbackPayload(BaseModel):
    features: dict
    label: int 
    transaction_id: Optional[str] = None



@app.get("/", tags=["Health"])
def root():
    return {"status": "online", "model": "LogisticRegression (River)"}


@app.post("/predict", tags=["Inference"])
def predict_transaction(tx: Transaction):

    x = tx.features
    try:
        proba_sgd = model_sgd.predict_proba_one(x)
        prob_sgd = proba_sgd.get(1, 0.0) if proba_sgd else 0.0
        
        proba_adam = model_adam.predict_proba_one(x)
        prob_adam = proba_adam.get(1, 0.0) if proba_adam else 0.0
    except Exception:
        prob_sgd = 0.0
        prob_adam = 0.0

    action = "BLOCK" if prob_sgd > 0.7 else "ALLOW"

    record = {
        "transaction_id": tx.transaction_id,
        "fraud_probability": round(prob_sgd, 6),
        "prob_sgd": round(prob_sgd, 6),
        "prob_adam": round(prob_adam, 6),
        "action": action,
        "timestamp": time.time(),
    }

    with log_lock:
        transaction_log.append(record)
        if len(transaction_log) > LOG_MAX:
            transaction_log.pop(0)
        stats["total"] += 1
        if action == "BLOCK":
            stats["blocked"] += 1

    return record


@app.post("/feedback", tags=["Learning"])
def feedback_label(payload: FeedbackPayload):

    x = payload.features
    y = payload.label

    try:
        proba_sgd = model_sgd.predict_proba_one(x)
        prob_sgd = proba_sgd.get(1, 0.0) if proba_sgd else 0.0
        pred_sgd = 1 if prob_sgd > 0.5 else 0
        model_sgd.learn_one(x, y)
        metric_sgd.update(y, prob_sgd)
        acc_sgd.update(y, pred_sgd)
        cm_sgd.update(y, pred_sgd)

        proba_adam = model_adam.predict_proba_one(x)
        prob_adam = proba_adam.get(1, 0.0) if proba_adam else 0.0
        pred_adam = 1 if prob_adam > 0.5 else 0
        model_adam.learn_one(x, y)
        metric_adam.update(y, prob_adam)
        acc_adam.update(y, pred_adam)
        cm_adam.update(y, pred_adam)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    with log_lock:
        if y == 1:
            stats["fraud_detected"] += 1
        stats["roc_auc_sgd"] = round(metric_sgd.get(), 6)
        stats["roc_auc_adam"] = round(metric_adam.get(), 6)
        stats["acc_sgd"] = round(acc_sgd.get(), 6)
        stats["acc_adam"] = round(acc_adam.get(), 6)
        
        # Get confusion matrix details (True/False Positives/Negatives)
        stats["cm_sgd"] = {
            "tp": cm_sgd.true_positives(1),
            "fp": cm_sgd.false_positives(1),
            "tn": cm_sgd.true_negatives(1),
            "fn": cm_sgd.false_negatives(1)
        }
        stats["cm_adam"] = {
            "tp": cm_adam.true_positives(1),
            "fp": cm_adam.false_positives(1),
            "tn": cm_adam.true_negatives(1),
            "fn": cm_adam.false_negatives(1)
        }

    return {
        "status": "model_updated",
        "roc_auc_sgd": stats["roc_auc_sgd"],
        "roc_auc_adam": stats["roc_auc_adam"],
        "transaction_id": payload.transaction_id,
    }


@app.get("/stats", tags=["Monitoring"])
def get_stats():
    with log_lock:
        return {
            **stats,
            "log_size": len(transaction_log),
        }


@app.get("/logs", tags=["Monitoring"])
def get_logs(limit: int = 50):
    with log_lock:
        return {"transactions": transaction_log[-limit:]}
