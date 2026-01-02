from flask import Flask, request, jsonify
from flask_cors import CORS
from urllib.parse import urlparse
import hashlib
import logging
import warnings
from datetime import datetime

try:
    from scripts.model_inferences import (
        XGBoostInference,
        RFInference,
        WordCNNInference,
        WordCNNLSTMInference,
        CharCNNInference,
        CharCNNLSTMInference,
        CNNHybridInference,
        ALBERTInference,
        MobileBERTInference,
        TinyLlamaInference,
    )
    from scripts.configs import FirebaseConfigs
    from scripts.firebase_client import FirebaseClient
except ModuleNotFoundError:
    from model_inferences import (
        XGBoostInference,
        RFInference,
        WordCNNInference,
        WordCNNLSTMInference,
        CharCNNInference,
        CharCNNLSTMInference,
        CNNHybridInference,
        ALBERTInference,
        MobileBERTInference,
        TinyLlamaInference,
    )
    from configs import FirebaseConfigs
    from firebase_client import FirebaseClient

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
warnings.filterwarnings("ignore")

INFERENCE_MAP = {
    "xgb": XGBoostInference,
    "rf": RFInference,
    "wordcnn": WordCNNInference,
    "wordcnn_lstm": WordCNNLSTMInference,
    "charcnn": CharCNNInference,
    "charcnn_lstm": CharCNNLSTMInference,
    "cnn_hybrid": CNNHybridInference,
    "albert": ALBERTInference,
    "mobile_bert": MobileBERTInference,
    "tiny_llama": TinyLlamaInference,
}

app = Flask(__name__)
CORS(app)
firebase_cfg = FirebaseConfigs()
firebase = FirebaseClient(
    firebase_cfg.info["cred_path"],
    firebase_cfg.info["project_id"],
    firebase_cfg.info["collections"],
)


def domain_hash(url: str) -> str:
    domain = urlparse(url).netloc.lower() or url
    return hashlib.sha256(domain.encode()).hexdigest()


def run_model(model_name: str, url: str, threshold: float):
    model_cls = INFERENCE_MAP.get(model_name)
    if not model_cls:
        raise ValueError(f"Unknown model '{model_name}'")
    model = model_cls(url)
    pred, proba = model.predict(threshold)
    return {"url": url, "pred": int(pred), "proba": float(proba), "model": model_name}


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    raw_url = data.get("url", "")
    model_name = data.get("model", "xgb").lower()
    threshold = float(data.get("threshold", 0.5))
    scan_mode = data.get("scan_mode", "full")

    if not raw_url:
        return jsonify({"error": "Missing 'url'"}), 400

    # 1. Giữ lại URL gốc
    original_url = raw_url.strip()

    # 2. Tạo URL để xử lý (Processed URL)
    processed_url = original_url
    if scan_mode == "domain":
        try:
            parsed = urlparse(original_url)
            processed_url = f"{parsed.scheme}://{parsed.netloc}"
        except Exception:
            pass

    # 3. Tạo Cache Key phân biệt mode
    cache_model_key = f"{model_name}_{scan_mode}"
    d_hash = domain_hash(processed_url)  # Hash dựa trên cái thực tế quét

    # --- Kiểm tra cache ---
    cached = firebase.get(d_hash, cache_model_key)
    if cached:
        cached["cached"] = True
        # Đảm bảo trả về đủ 2 url để hiển thị
        cached["url"] = processed_url
        cached["original_url"] = original_url
        return jsonify(cached), 200

    # --- Dự đoán mới ---
    try:
        # Chạy model với URL đã xử lý
        result = run_model(model_name, processed_url, threshold)

        # Bổ sung thông tin URL gốc vào kết quả
        result["original_url"] = original_url

        # Lưu cache
        firebase.save(d_hash, cache_model_key, result)

        result["cached"] = False
        return jsonify(result), 200
    except Exception as e:
        logging.exception("Prediction error")
        return jsonify({"error": str(e)}), 500


@app.route("/healthz")
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/api/logs", methods=["GET"])
def get_logs():
    try:
        # Nhận tham số limit từ URL. Nếu client gửi 0 hoặc không gửi, ta hiểu là lấy hết.
        limit_param = request.args.get("limit", default=0, type=int)

        # Nếu limit = 0 -> truyền None vào hàm firebase để lấy hết
        limit = limit_param if limit_param > 0 else None

        logs = firebase.get_all_logs(limit=limit)
        return jsonify(logs), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/logs/<doc_id>", methods=["PUT"])
def update_log_entry(doc_id):
    """API để sửa nhãn hoặc cập nhật trạng thái."""
    try:
        data = request.get_json()
        allowed_updates = {}
        if "pred" in data:
            allowed_updates["pred"] = int(data["pred"])
        if "is_verified" in data:
            allowed_updates["is_verified"] = bool(data["is_verified"])

        if not allowed_updates:
            return jsonify({"error": "No valid fields to update"}), 400

        allowed_updates["updated_at"] = datetime.utcnow().isoformat()

        success = firebase.update(doc_id, allowed_updates)
        if success:
            return (
                jsonify(
                    {"status": "updated", "id": doc_id, "updates": allowed_updates}
                ),
                200,
            )
        else:
            return jsonify({"error": "Failed to update in Firestore"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/logs/<doc_id>", methods=["DELETE"])
def delete_log(doc_id):
    try:
        firebase.col.document(doc_id).delete()
        return jsonify({"status": "deleted", "id": doc_id}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
