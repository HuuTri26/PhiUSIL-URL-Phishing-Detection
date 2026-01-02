const API_BASE = "http://127.0.0.1:8000/predict";

// giá trị default nếu chưa cấu hình
const DEFAULT_MODEL = "rf";
const DEFAULT_THRESHOLD = 0.6;
const DEFAULT_SCAN_MODE = "full"; // "full" hoặc "domain"
// đọc cấu hình model/threshold hiện tại
function getCurrentConfig() {
  return new Promise((resolve) => {
    chrome.storage.sync.get(
      ["phiusiil_model", "phiusiil_threshold", "phiusiil_scan_mode"],
      (data) => {
        let model = data.phiusiil_model || DEFAULT_MODEL;
        let scan_mode = data.phiusiil_scan_mode || DEFAULT_SCAN_MODE; // Lấy mode
        let threshold =
          typeof data.phiusiil_threshold === "number"
            ? data.phiusiil_threshold
            : DEFAULT_THRESHOLD;

        // Clamp threshold
        if (Number.isNaN(threshold)) threshold = DEFAULT_THRESHOLD;
        threshold = Math.min(Math.max(threshold, 0), 1);

        resolve({ model, threshold, scan_mode });
      }
    );
  });
}

function setBadge(tabId, { text, color }) {
  chrome.action.setBadgeText({ tabId, text });
  chrome.action.setBadgeBackgroundColor({ tabId, color });
}

// lưu kết quả theo tab để popup hien thi  lại
function storeResult(tabId, result) {
  const key = "phiusiil_result_" + tabId;
  chrome.storage.local.set({ [key]: result });
}

// scan 1 tab cụ thể
async function scanTab(tabId, url) {
  if (!url || !/^https?:\/\//i.test(url)) {
    // không scan các scheme khác
    setBadge(tabId, { text: "", color: [0, 0, 0, 0] });
    chrome.tabs.sendMessage(tabId, { type: "phiusiil_clear" }).catch(() => {});
    storeResult(tabId, null);
    return;
  }

  const { model, threshold, scan_mode } = await getCurrentConfig();

  // badge trạng thái đang quét
  setBadge(tabId, { text: "…", color: [38, 50, 56, 255] });

  const payload = { url, model, threshold, scan_mode };

  try {
    const res = await fetch(API_BASE, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`HTTP ${res.status}: ${text}`);
    }

    const data = await res.json();

    const pred = Number(data.pred); // 0 = phishing, 1 = benign
    const proba = Number(data.proba) || 0;
    const isPhishing = pred === 0;

    // set badge
    if (isPhishing) {
      setBadge(tabId, { text: "P", color: [183, 28, 28, 255] }); // đỏ
    } else {
      setBadge(tabId, { text: "OK", color: [46, 125, 50, 255] }); // xanh
    }

    const result = {
      ...data,
      threshold,
      isPhishing
    };
    storeResult(tabId, result);

    // gửi cho content script highlight trang
    chrome.tabs
      .sendMessage(tabId, {
        type: "phiusiil_scan_result",
        result
      })
      .catch(() => {});
  } catch (err) {
    console.error("Scan error:", err);
    setBadge(tabId, { text: "ERR", color: [117, 117, 117, 255] });
    storeResult(tabId, { error: err.message });

    chrome.tabs
      .sendMessage(tabId, {
        type: "phiusiil_error",
        error: err.message
      })
      .catch(() => {});
  }
}

// lắng nghe khi tab load xong
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === "complete" && tab.active) {
    scanTab(tabId, tab.url);
  }
});

// lắng nghe khi chuyển tab
chrome.tabs.onActivated.addListener((activeInfo) => {
  chrome.tabs.get(activeInfo.tabId, (tab) => {
    if (chrome.runtime.lastError || !tab) return;
    scanTab(activeInfo.tabId, tab.url);
  });
});

// cho popup yêu cầu rescan / lấy kết quả hiện tại
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "phiusiil_get_result") {
    const tabId = message.tabId;
    const key = "phiusiil_result_" + tabId;
    chrome.storage.local.get([key], (data) => {
      sendResponse({ result: data[key] || null });
    });
    return true; // async
  }

  if (message.type === "phiusiil_rescan") {
    const tabId = message.tabId;
    chrome.tabs.get(tabId, (tab) => {
      if (chrome.runtime.lastError || !tab) {
        sendResponse({ ok: false });
        return;
      }
      scanTab(tabId, tab.url).then(() => sendResponse({ ok: true }));
    });
    return true; // async
  }
});
