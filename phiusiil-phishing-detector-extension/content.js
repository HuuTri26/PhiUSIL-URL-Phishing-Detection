let overlayEl = null;
let bannerEl = null;

function removeHighlight() {
  if (overlayEl && overlayEl.parentNode) {
    overlayEl.parentNode.removeChild(overlayEl);
  }
  if (bannerEl && bannerEl.parentNode) {
    bannerEl.parentNode.removeChild(bannerEl);
  }
  overlayEl = null;
  bannerEl = null;
}

function createHighlight(result) {
  removeHighlight();

  if (!result || !result.isPhishing) {
    return; // chỉ highlight khi phishing
  }

  // lớp overlayy mờ đỏ
  overlayEl = document.createElement("div");
  overlayEl.style.position = "fixed";
  overlayEl.style.top = "0";
  overlayEl.style.left = "0";
  overlayEl.style.right = "0";
  overlayEl.style.bottom = "0";
  overlayEl.style.pointerEvents = "none";
  overlayEl.style.background =
    "rgba(183, 28, 28, 0.18)"; // lớp mờ đỏ
  overlayEl.style.zIndex = "2147483640";
  document.documentElement.appendChild(overlayEl);

  // banner cảnh báo
  bannerEl = document.createElement("div");
  bannerEl.style.position = "fixed";
  bannerEl.style.top = "0";
  bannerEl.style.left = "0";
  bannerEl.style.right = "0";
  bannerEl.style.padding = "8px 12px";
  bannerEl.style.background = "#b71c1c";
  bannerEl.style.color = "#ffebee";
  bannerEl.style.fontFamily =
    'system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif';
  bannerEl.style.fontSize = "13px";
  bannerEl.style.display = "flex";
  bannerEl.style.alignItems = "center";
  bannerEl.style.justifyContent = "space-between";
  bannerEl.style.zIndex = "2147483641";
  bannerEl.style.boxShadow = "0 2px 6px rgba(0,0,0,0.4)";

  const left = document.createElement("div");
  left.textContent =
    " Cảnh báo: URL có khả năng là phishing (label = 0).";

  const right = document.createElement("div");
  right.style.fontSize = "11px";
  right.style.opacity = "0.9";
  const p = Number(result.proba || 0);
  const modelName = result.model || "?";
  right.textContent = `Model: ${modelName} • p = ${p.toFixed(3)}`;

  bannerEl.appendChild(left);
  bannerEl.appendChild(right);

  document.documentElement.appendChild(bannerEl);
}

chrome.runtime.onMessage.addListener((message) => {
  if (message.type === "phiusiil_scan_result") {
    createHighlight(message.result);
  } else if (message.type === "phiusiil_clear") {
    removeHighlight();
  } else if (message.type === "phiusiil_error") {
    // lỗi API → xoá highlight (tránh gây hiểu nhầm)
    removeHighlight();
  }
});
