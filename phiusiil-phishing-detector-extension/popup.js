const els = {};

document.addEventListener("DOMContentLoaded", async () => {
  els.currentUrl = document.getElementById("current-url");
  els.modelSelect = document.getElementById("model-select");
  els.thresholdInput = document.getElementById("threshold-input");
  els.rescanBtn = document.getElementById("rescan-btn");

  els.status = document.getElementById("status");
  els.result = document.getElementById("result");
  els.error = document.getElementById("error");

  els.resModel = document.getElementById("res-model");
  els.resPred = document.getElementById("res-pred");
  els.resProba = document.getElementById("res-proba");
  els.resThreshold = document.getElementById("res-threshold");
  els.resCached = document.getElementById("res-cached");
  els.resCreated = document.getElementById("res-created");
  els.resUrl = document.getElementById("res-url");
  els.resOriginalUrl = document.getElementById("res-original-url");
  // load config từ storage
  chrome.storage.sync.get(["phiusiil_model", "phiusiil_threshold"], (data) => {
    if (data.phiusiil_model) {
      els.modelSelect.value = data.phiusiil_model;
    }
    if (typeof data.phiusiil_threshold === "number") {
      els.thresholdInput.value = data.phiusiil_threshold.toFixed(2);
    }
  });

  // Khi load popup
  chrome.storage.sync.get(["phiusiil_scan_mode"], (data) => {
    document.getElementById("scanMode").value = data.phiusiil_scan_mode || "full";
  });

    // Khi thay đổi
  document.getElementById("scanMode").addEventListener("change", (e) => {
  chrome.storage.sync.set({ phiusiil_scan_mode: e.target.value });
    // Gửi message yêu cầu rescan nếu cần
  });

  els.modelSelect.addEventListener("change", saveConfig);
  els.thresholdInput.addEventListener("change", saveConfig);

  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const tabId = tab ? tab.id : null;
  const url = tab && tab.url ? tab.url : "";
  els.currentUrl.textContent = url || "(Không lấy được URL)";
  els.resUrl.textContent = url;

  if (tabId == null) {
    setError("Không lấy được thông tin tab hiện tại.");
    return;
  }

  // lấy kết quả auto-scan từ background
  requestCurrentResult(tabId);

  // nút quét lạ
  els.rescanBtn.addEventListener("click", () => {
    setStatus("loading", "Đang quét lại...");
    chrome.runtime.sendMessage(
      { type: "phiusiil_rescan", tabId },
      () => {
        // sau khi rescan xong, lấy lại kết quả
        setTimeout(() => requestCurrentResult(tabId), 300);
      }
    );
  });
});

function saveConfig() {
  let threshold = parseFloat(els.thresholdInput.value);
  if (Number.isNaN(threshold)) threshold = 0.6;
  threshold = Math.min(Math.max(threshold, 0), 1);
  els.thresholdInput.value = threshold.toFixed(2);

  chrome.storage.sync.set({
    phiusiil_model: els.modelSelect.value,
    phiusiil_threshold: threshold
  });
}

function setStatus(type, message) {
  els.status.classList.remove("hidden", "safe", "phishing", "loading");
  els.status.textContent = message;

  if (type === "safe") els.status.classList.add("safe");
  else if (type === "phishing") els.status.classList.add("phishing");
  else if (type === "loading") els.status.classList.add("loading");
}

function clearMessages() {
  els.error.classList.add("hidden");
  els.error.textContent = "";
  els.result.classList.add("hidden");
}

function setError(msg) {
  clearMessages();
  els.error.textContent = msg;
  els.error.classList.remove("hidden");
}

function requestCurrentResult(tabId) {
  clearMessages();
  setStatus("loading", "Đang lấy kết quả auto-scan...");

  chrome.runtime.sendMessage(
    { type: "phiusiil_get_result", tabId },
    (response) => {
      if (!response || !response.result) {
        setStatus("loading", "Chưa có kết quả (đang quét hoặc lỗi).");
        return;
      }

      const result = response.result;
      if (result.error) {
        setError("Lỗi API: " + result.error);
        setStatus("phishing", "Không đánh giá được – kiểm tra API.");
        return;
      }

      showResult(result);
    }
  );
}

function showResult(result) {
  const pred = Number(result.pred); // 0 = phishing, 1 = benign
  const proba = Number(result.proba) || 0;
  const threshold = Number(result.threshold) || 0.6;
  const cached = !!result.cached;
  const model = result.model || "?";
  const createdAt = result.created_at || "";
  const url = result.url || "";
  const originalUrl = result.original_url || url; // Fallback về url nếu không có

  const isPhishing = pred === 0;

  if (isPhishing) {
    setStatus(
      "phishing",
      `️ Phishing / URL đáng ngờ (label = 0, p = ${proba.toFixed(3)})`
    );
  } else {
    setStatus(
      "safe",
      ` Benign / Có vẻ an toàn (label = 1, p = ${proba.toFixed(3)})`
    );
  }

  // Nếu 2 URL giống nhau (chế độ Full) thì có thể ẩn bớt 1 dòng cho gọn (Tùy chọn)
  if (url === originalUrl) {
      els.resOriginalUrl.parentElement.style.display = 'none';
      els.resUrl.previousElementSibling.textContent = "URL:"; // Đổi lại nhãn
  } else {
      els.resOriginalUrl.parentElement.style.display = 'flex';
      els.resUrl.previousElementSibling.textContent = "URL Đã quét:";
  }

  els.resModel.textContent = model;
  els.resPred.textContent = isPhishing
    ? "0 (phishing / suspicious)"
    : "1 (benign / likely safe)";
  els.resProba.textContent = proba.toFixed(4);
  els.resThreshold.textContent = threshold.toFixed(2);
  els.resCached.textContent = cached ? "Yes (dùng cache Firestore)" : "No (dự đoán mới)";

  if (createdAt) {
    try {
      const dt = new Date(createdAt);
      els.resCreated.textContent = dt.toLocaleString();
    } catch {
      els.resCreated.textContent = createdAt;
    }
  } else {
    els.resCreated.textContent = "(không có)";
  }

  els.resUrl.textContent = url;
  els.result.classList.remove("hidden");
}
