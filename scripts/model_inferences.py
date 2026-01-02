import os
import threading
import logging
import warnings
import time
from importlib import reload

import numpy as np
import joblib
import torch
import re

from urllib.parse import urlparse
from gensim.models import Word2Vec
from xgboost import XGBClassifier
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from peft import PeftModel
try:
    from scripts.model_downloader import ModelDownloader
except ModuleNotFoundError:
    from model_downloader import ModelDownloader


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loader = ModelDownloader()


class W2VecInference:
    """PhiUSIIL Word2Vec"""
    _instance = None
    _lock = threading.Lock()
    _model = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(W2VecInference, cls).__new__(cls)
        return cls._instance

    def __init__(self, url, dim=50):
        self._dim = dim
        self._tokens = []
        self._url = url

    @classmethod
    def _load_model(cls):
        """Load Word2Vec model từ file config."""
        loader.download_single("WORD2VEC")
        model_path = loader.cfg.paths["WORD2VEC"]["full_path"]
        model = Word2Vec.load(model_path)
        logging.info(f"✅ Word2Vec model loaded successfully from {model_path}")
        return model

    @classmethod
    def _get_model(cls, reload=False):
        """Lấy instance Word2Vec (cache)."""
        with cls._lock:
            if cls._model is None or reload:
                cls._model = cls._load_model()
            return cls._model

    def _preprocess(self):
        """Chuyển URL thành danh sách token hợp lệ."""
        if not self._url or not isinstance(self._url, (str, bytes)):
            raise ValueError("URL phải là chuỗi hợp lệ")

        self._tokens.clear()
        self._url = self._url.decode('utf-8') if isinstance(self._url, bytes) else self._url.lower()
        parsed = urlparse(self._url)

        # Tách domain, path, query
        domain_parts = parsed.netloc.split('.') if parsed.netloc else []
        path_parts = [p for p in parsed.path.split('/') if p]
        query_parts = parsed.query.split('&') if parsed.query else []

        tokens = domain_parts + path_parts + query_parts
        # Lọc ký tự đặc biệt
        clean_tokens = [re.sub(r'[^a-z0-9\-/.=]', '', t) for t in tokens if t]
        self._tokens.extend(clean_tokens)

        return self._tokens

    def get_vector(self):
        """Trích vector trung bình từ các token trong URL."""
        tokens = self._preprocess()
        vectors = [self._get_model().wv[token] for token in tokens if token in self._get_model().wv]
        if vectors:
            vec = np.mean(vectors, axis=0)
        else:
            vec = np.zeros(self._dim)
        return vec


class CharTokenizerInference:
    """PhiUSIIL CharTokenizer"""
    _instance = None
    _lock = threading.Lock()
    _tokenizer = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(CharTokenizerInference, cls).__new__(cls)
        return cls._instance

    def __init__(self, url, max_len=200):
        self._max_len = max_len
        self._url = url.lower().strip()
        self._url = re.sub(r'\s+', '', self._url)

    @classmethod
    def _load_tokenizer(cls):
        """Load CharTokenizer từ file config."""
        loader.download_single("CHAR_TOKENIZER")
        tokenizer_path = loader.cfg.paths["CHAR_TOKENIZER"]["full_path"]
        tokenizer = joblib.load(tokenizer_path)
        logging.info(f"✅ CharTokenizer loaded successfully from {tokenizer_path}")
        return tokenizer

    @classmethod
    def get_tokenizer(cls, reload=False):
        """Lấy instance CharTokenizer (cache)."""
        with cls._lock:
            if cls._tokenizer is None or reload:
                cls._tokenizer = cls._load_tokenizer()
        return cls._tokenizer

    def get_padded_sequence(self):
        """Chuyển chuỗi url thành padded sequence"""
        seq = self.get_tokenizer().texts_to_sequences([self._url])[0]
        padded_seq = pad_sequences([seq], maxlen=self._max_len, padding="post", truncating="post")
        return padded_seq

    def vocab_size(self):
        return len(self.get_tokenizer().word_index) + 1


class XGBoostInference:
    """PhiUSIIL XGBoost"""
    _instance = None
    _lock = threading.Lock()
    _model = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(XGBoostInference, cls).__new__(cls)
        return cls._instance

    def __init__(self, url):
        self._url = url

    @classmethod
    def _load_model(cls):
        """Load XGBoost model từ file config."""
        loader.download_single("XGB")
        model_path = loader.cfg.paths["XGB"]["full_path"]
        model = XGBClassifier()
        model.load_model(model_path)
        logging.info(f"✅ XGBoost model loaded successfully from {model_path}")
        return model

    @classmethod
    def _get_model(cls, reload=False):
        """Lấy instance XGB (cache)."""
        with cls._lock:
            if cls._model is None or reload:
                cls._model = cls._load_model()
            return cls._model

    def predict(self, threshold=0.5):
        w2v = W2VecInference(self._url)
        vector = w2v.get_vector().reshape(1, -1)
        proba = self._get_model().predict_proba(vector)[0][1]
        pred = int(proba >= threshold)
        return pred, proba


class RFInference:
    """PhiUSIIL Random Forest"""
    _instance = None
    _lock = threading.Lock()
    _model = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(RFInference, cls).__new__(cls)
        return cls._instance

    def __init__(self, url):
        self._url = url

    @classmethod
    def _load_model(cls):
        """Load Random Forest model từ file config."""
        loader.download_single("RF")
        model_path = loader.cfg.paths["RF"]["full_path"]
        model = joblib.load(model_path)
        logging.info(f"✅ Random Forest model loaded successfully from {model_path}")
        return model

    @classmethod
    def _get_model(cls, reload=False):
        """Lấy instance Random Forest (cache)."""
        with cls._lock:
            if cls._model is None or reload:
                cls._model = cls._load_model()
            return cls._model

    def predict(self, threshold=0.5):
        w2v = W2VecInference(self._url)
        vector = w2v.get_vector().reshape(1, -1)
        proba = self._get_model().predict_proba(vector)[0][1]
        pred = int(proba >= threshold)
        return pred, proba


class WordCNNInference:
    """PhiUSIIL WordCNN"""
    _instance = None
    _lock = threading.Lock()
    _model = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(WordCNNInference, cls).__new__(cls)
        return cls._instance

    def __init__(self, url):
        self._url = url

    @classmethod
    def _load_model(cls):
        """Load WordCNN model từ file config."""
        loader.download_single("WORD_CNN")
        model_path = loader.cfg.paths["WORD_CNN"]["full_path"]
        model = load_model(model_path)
        logging.info(f"✅ WordCNN model loaded successfully from {model_path}")
        return model

    @classmethod
    def _get_model(cls):
        """Lấy instance WordCNN (cache)."""
        with cls._lock:
            if cls._model is None or reload:
                cls._model = cls._load_model()
            return cls._model

    def predict(self, threshold=0.5):
        w2v = W2VecInference(self._url)
        vector = w2v.get_vector().reshape(1, -1)
        preds = self._get_model().predict(vector, verbose=0)
        proba = float(preds[0][0])

        pred = int(proba >= threshold)
        return pred, proba


class WordCNNLSTMInference:
    """PhiUSIIL WordCNN_LSTM"""
    _instance = None
    _lock = threading.Lock()
    _model = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(WordCNNLSTMInference, cls).__new__(cls)
        return cls._instance

    def __init__(self, url):
        self._url = url

    @classmethod
    def _load_model(cls):
        """Load WordCNN model từ file config."""
        loader.download_single("WORD_CNN_LSTM")
        model_path = loader.cfg.paths["WORD_CNN_LSTM"]["full_path"]
        model = load_model(model_path)
        logging.info(f"✅ WordCNN_LSTM model loaded successfully from {model_path}")
        return model

    @classmethod
    def _get_model(cls):
        """Lấy instance WordCNN_LSTM (cache)."""
        with cls._lock:
            if cls._model is None or reload:
                cls._model = cls._load_model()
            return cls._model

    def predict(self, threshold=0.5):
        w2v = W2VecInference(self._url)
        vector = w2v.get_vector().reshape(1, -1)
        preds = self._get_model().predict(vector, verbose=0)
        proba = float(preds[0][0])

        pred = int(proba >= threshold)
        return pred, proba


class CharCNNInference:
    """PhiUSIIL CharCNN"""
    _instance = None
    _lock = threading.Lock()
    _model = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(CharCNNInference, cls).__new__(cls)
        return cls._instance

    def __init__(self, url):
        self._url = url

    @classmethod
    def _load_model(cls):
        """Load CharCNN model từ file config."""
        loader.download_single("CHAR_CNN")
        model_path = loader.cfg.paths["CHAR_CNN"]["full_path"]
        model = load_model(model_path)
        logging.info(f"✅ CharCNN model loaded successfully from {model_path}")
        return model

    @classmethod
    def _get_model(cls):
        """Lấy instance CharCNN (cache)."""
        with cls._lock:
            if cls._model is None or reload:
                cls._model = cls._load_model()
            return cls._model

    def predict(self, threshold=0.5):
        tokenizer = CharTokenizerInference(self._url)
        seq = tokenizer.get_padded_sequence()
        preds = self._get_model().predict(seq, verbose=0)
        proba = float(preds[0][0])

        pred = int(proba >= threshold)
        return pred, proba


class CharCNNLSTMInference:
    """PhiUSIIL CharCNN-LSTM"""
    _instance = None
    _lock = threading.Lock()
    _model = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(CharCNNLSTMInference, cls).__new__(cls)
        return cls._instance

    def __init__(self, url):
        self._url = url

    @classmethod
    def _load_model(cls):
        """Load CharCNN_LSTM model từ file config."""
        loader.download_single("CHAR_CNN_LSTM")
        model_path = loader.cfg.paths["CHAR_CNN_LSTM"]["full_path"]
        model = load_model(model_path)
        logging.info(f"✅ CharCNN_LSTM model loaded successfully from {model_path}")
        return model

    @classmethod
    def _get_model(cls):
        """Lấy instance CharCNN_LSMT (cache)."""
        with cls._lock:
            if cls._model is None or reload:
                cls._model = cls._load_model()
            return cls._model

    def predict(self, threshold=0.5):
        tokenizer = CharTokenizerInference(self._url)
        seq = tokenizer.get_padded_sequence()
        preds = self._get_model().predict(seq, verbose=0)
        proba = float(preds[0][0])

        pred = int(proba >= threshold)
        return pred, proba


class CNNHybridInference:
    """PhiUSIIL CNN Hybrid"""
    _instance = None
    _lock = threading.Lock()
    _model = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(CNNHybridInference, cls).__new__(cls)
        return cls._instance

    def __init__(self, url):
        self._url = url

    @classmethod
    def _load_model(cls):
        """Load CNN_Hybrid model từ file config."""
        loader.download_single("CNN_HYBRID")
        model_path = loader.cfg.paths["CNN_HYBRID"]["full_path"]
        model = load_model(model_path)
        logging.info(f"✅ CNN_HYBRID model loaded successfully from {model_path}")
        return model

    @classmethod
    def _get_model(cls):
        """Lấy instance CNN_HYBRID (cache)."""
        with cls._lock:
            if cls._model is None or reload:
                cls._model = cls._load_model()
            return cls._model

    def predict(self, threshold=0.5):
        tokenizer = CharTokenizerInference(self._url)
        w2v = W2VecInference(self._url)
        seq = tokenizer.get_padded_sequence()
        vec = w2v.get_vector().reshape(1, -1)
        preds = self._get_model().predict([seq, vec], verbose=0)
        proba = float(preds[0][0])

        pred = int(proba >= threshold)
        return pred, proba


class ALBERTInference:
    """PhiUSIIL ALBERT"""
    _instance = None
    _lock = threading.Lock()
    _model = None
    _tokenizer = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ALBERTInference, cls).__new__(cls)
        return cls._instance

    def __init__(self, url, max_len=128):
        self._url = url
        self._max_len = max_len

    @classmethod
    def _load_tokenizer(cls):
        """Load ALBERT's tokenizer từ file config."""
        loader.download_single("ALBERT")
        tokenizer_path = loader.cfg.paths["ALBERT"]["full_path"]
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        logging.info(f"✅ ALBERT's tokenizer loaded successfully from {tokenizer_path}")
        return tokenizer

    @classmethod
    def _load_model(cls):
        """Load ALBERT model từ file config."""
        loader.download_single("ALBERT")
        model_path = loader.cfg.paths["ALBERT"]["full_path"]
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        logging.info(f"✅ ALBERT model loaded successfully from {model_path}")
        return model

    @classmethod
    def _get_tokenizer(cls, reload=False):
        """Lấy instance ALBERT (cache/tokenizer)."""
        with cls._lock:
            if cls._tokenizer is None or reload:
                cls._tokenizer = cls._load_tokenizer()
            return cls._tokenizer

    @classmethod
    def _get_model(cls, reload=False):
        """Lấy instance ALBERT (cache/model)."""
        with cls._lock:
            if cls._model is None or reload:
                cls._model = cls._load_model()
            return cls._model

    def predict(self, threshold=0.5):
        """Trả về (pred_label:int, probs:tensor[1,2])."""
        tokenizer = self._get_tokenizer()
        model = self._get_model()

        inputs = tokenizer(self._url, truncation=True, padding=True,
                           max_length=self._max_len, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
            pred = torch.argmax(probs).item()
            prob = probs[0].item() if pred == 0 else probs[1].item()

        return pred, prob


class MobileBERTInference:
    """PhiUSIIL MobileBERT"""
    _instance = None
    _lock = threading.Lock()
    _model = None
    _tokenizer = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(MobileBERTInference, cls).__new__(cls)
        return cls._instance

    def __init__(self, url, max_len=128):
        self._url = url
        self._max_len = max_len

    @classmethod
    def _load_tokenizer(cls):
        """Load MOBILE_BERT's tokenizer từ file config."""
        loader.download_single("MOBILE_BERT")
        tokenizer_path = loader.cfg.paths["MOBILE_BERT"]["full_path"]
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        logging.info(f"✅ MOBILE_BERT's tokenizer loaded successfully from {tokenizer_path}")
        return tokenizer

    @classmethod
    def _load_model(cls):
        """Load MOBILE_BERT model từ file config."""
        loader.download_single("MOBILE_BERT")
        model_path = loader.cfg.paths["MOBILE_BERT"]["full_path"]
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        logging.info(f"✅ MOBILE_BERT model loaded successfully from {model_path}")
        return model

    @classmethod
    def _get_tokenizer(cls, reload=False):
        """Lấy instance MOBILE_BERT (cache/tokenizer)."""
        with cls._lock:
            if cls._tokenizer is None or reload:
                cls._tokenizer = cls._load_tokenizer()
            return cls._tokenizer

    @classmethod
    def _get_model(cls, reload=False):
        """Lấy instance MOBILE_BERT (cache/model)."""
        with cls._lock:
            if cls._model is None or reload:
                cls._model = cls._load_model()
            return cls._model

    def predict(self, threshold=0.5):
        """Trả về (pred_label:int, probs:tensor[1,2])."""
        tokenizer = self._get_tokenizer()
        model = self._get_model()

        inputs = tokenizer(self._url, truncation=True, padding=True,
                           max_length=self._max_len, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
            pred = torch.argmax(probs).item()
            prob = probs[0].item() if pred == 0 else probs[1].item()

        return pred, prob


class TinyLlamaInference:
    """PhiUSIIL TinyLlama LoRA"""
    _instance = None
    _lock = threading.Lock()
    _model = None
    _tokenizer = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(TinyLlamaInference, cls).__new__(cls)
        return cls._instance

    def __init__(self, url, max_len=256):
        self._url = url
        self._max_len = max_len

    @classmethod
    def _load_tokenizer(cls):
        """Load tokenizer TinyLlama + LoRA checkpoint."""
        loader.download_single("TINY_LLAMA_LORA")
        model_path = loader.cfg.paths["TINY_LLAMA_LORA"]["full_path"]

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token

        logging.info(f"✅ TinyLlama tokenizer loaded from {model_path}")
        return tokenizer

    @classmethod
    def _load_model(cls):
        """Load TinyLlama LoRA model."""
        loader.download_single("TINY_LLAMA_BASE")
        base_model_path = loader.cfg.paths["TINY_LLAMA_BASE"]["full_path"]
        loader.download_single("TINY_LLAMA_LORA")
        lora_path = loader.cfg.paths["TINY_LLAMA_LORA"]["full_path"]

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        logging.info(f"✅ TinyLlama model loaded from {base_model_path}")

        # Load LoRA adapter
        model = PeftModel.from_pretrained(
            base_model,
            lora_path,
            torch_dtype=torch.float16
        )
        logging.info(f"✅ TinyLlama LoRA loaded from {lora_path}")

        model = model.to(device)
        return model

    @classmethod
    def _get_tokenizer(cls, reload=False):
        with cls._lock:
            if cls._tokenizer is None or reload:
                cls._tokenizer = cls._load_tokenizer()
            return cls._tokenizer

    @classmethod
    def _get_model(cls, reload=False):
        with cls._lock:
            if cls._model is None or reload:
                cls._model = cls._load_model()
            return cls._model

    def predict(self, threshold=0.5):
        tokenizer = self._get_tokenizer()
        model = self._get_model()

        # Chat template đúng lúc training
        messages = [
            {"role": "system",
             "content": "You are a cybersecurity assistant that classifies URLs as phishing or benign."},
            {"role": "user", "content": f"Classify the following URL:\n{self._url}"}
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                temperature=0.0,
                top_p=0.0,
                repetition_penalty=1.0
            )

        # Decode phần model generate
        answer = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        ).strip().lower()
        # Clean output
        answer = answer.replace(".", "").replace("?", "").strip()

        if answer.startswith("ph"):
            return 0, 0.95
        if answer.startswith("ben"):
            return 1, 0.95

        # fallback
        return 0, 0.0
