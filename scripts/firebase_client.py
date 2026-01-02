import os
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, firestore


class FirebaseClient:
    """Đóng gói Firestore cho lưu & truy vấn kết quả dự đoán."""

    def __init__(self, creds_path: str, project_id: str, collection: str = "predictions", ttl_days: int = 30):
        if not firebase_admin._apps:
            cred = credentials.Certificate(creds_path)
            firebase_admin.initialize_app(cred, {"projectId": project_id})
        self.db = firestore.client()
        self.col = self.db.collection(collection)
        self.ttl = timedelta(days=ttl_days)

    def _key(self, domain_hash: str, model: str) -> str:
        return f"{domain_hash}:{model}"

    def get(self, domain_hash: str, model: str):
        doc = self.col.document(self._key(domain_hash, model)).get()
        if not doc.exists:
            return None
        data = doc.to_dict()
        created = datetime.fromisoformat(data.get("created_at"))
        if datetime.utcnow() - created > self.ttl:
            self.col.document(self._key(domain_hash, model)).delete()
            return None
        return data

    def save(self, domain_hash: str, model: str, payload: dict):
        payload["created_at"] = datetime.utcnow().isoformat()
        self.col.document(self._key(domain_hash, model)).set(payload)

    def get_all_logs(self, limit=None):
        """Lấy danh sách log. Nếu limit=None thì lấy tất cả."""
        try:
            query = self.col.order_by("created_at", direction=firestore.Query.DESCENDING)

            # Chỉ giới hạn nếu có tham số limit
            if limit:
                query = query.limit(limit)

            docs = query.stream()
            results = []
            for doc in docs:
                data = doc.to_dict()
                data["id"] = doc.id
                results.append(data)
            return results
        except Exception as e:
            print(f"Error fetching logs: {e}")
            return []

    def update(self, doc_id: str, updates: dict):
        """Cập nhật một số trường cụ thể cho document."""
        try:
            self.col.document(doc_id).update(updates)
            return True
        except Exception as e:
            print(f"Error updating doc {doc_id}: {e}")
            return False