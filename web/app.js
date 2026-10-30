const pilotRows = [
  { model: "EfficientNet-B0", arm: "Original", accuracy: 0.9505356, macro_f1: 0.9547175, roc_auc: 0.9941264, test_n: 3174 },
  { model: "EfficientNet-B0", arm: "Fixed · CLAHE", accuracy: 0.9543163, macro_f1: 0.9575529, roc_auc: 0.9939662, test_n: 3174 },
  { model: "EfficientNet-B0", arm: "Fixed · Denoise", accuracy: 0.9502205, macro_f1: 0.9564763, roc_auc: 0.9939531, test_n: 3174 },
  { model: "EfficientNet-B0", arm: "Fixed · CLAHE + Denoise", accuracy: 0.9495904, macro_f1: 0.9558072, roc_auc: 0.9939486, test_n: 3174 },
  { model: "EfficientNet-B0", arm: "Fixed · Wiener", accuracy: 0.9473850, macro_f1: 0.9517671, roc_auc: 0.9937248, test_n: 3174 },
  { model: "EfficientNet-B0", arm: "Fixed · CLAHE + Wiener", accuracy: 0.9426591, macro_f1: 0.9483968, roc_auc: 0.9923272, test_n: 3174 },
  { model: "EfficientNet-B0", arm: "Adaptive", accuracy: 0.9511657, macro_f1: 0.9545197, roc_auc: 0.9936352, test_n: 3174 },
];
const stressRows = [
  { label: "Underexposure", winner: "Fixed CLAHE", accuracy: 93.67 },
  { label: "Gaussian blur", winner: "Fixed CLAHE + Wiener", accuracy: 92.03 },
  { label: "Gaussian noise", winner: "Fixed Wiener", accuracy: 20.29 },
];

function pct(value, digits = 2) { return `${(Number(value) * 100).toFixed(digits)}%`; }
function renderTable(rows) {
  document.querySelector("#metrics-body").innerHTML = rows.map(row => `<tr><td>${escapeHtml(row.arm)}</td><td>${pct(row.accuracy)}</td><td>${Number(row.macro_f1).toFixed(4)}</td><td>${Number(row.roc_auc).toFixed(4)}</td><td>${row.test_n || "—"}</td></tr>`).join("");
}
function renderChart(rows) {
  const chart = document.querySelector("#clean-chart");
  chart.innerHTML = "";
  const colors = ["original", "fixed", "fixed", "fixed", "fixed", "fixed", "adaptive"];
  rows.forEach((row, index) => {
    const value = Number(row.accuracy) * 100;
    const column = document.createElement("div");
    column.className = "chart-column";
    const label = row.arm.replace("Fixed · ", "").replace("bilateral denoising", "Denoise").replace("CLAHE + bilateral denoising", "CLAHE + denoise");
    column.innerHTML = `<div class="chart-bar ${colors[index % colors.length]} palette-${index}" style="height:${Math.max(6, Math.min(100, (value - 90) * 10))}%"><b>${value.toFixed(2)}%</b></div><span class="chart-label">${escapeHtml(label)}</span>`;
    chart.appendChild(column);
  });
}
function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, character => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[character]);
}
function formatMs(value) {
  const ms = Number(value);
  return Number.isFinite(ms) ? `${ms < 10 ? ms.toFixed(2) : ms.toFixed(1)} ms` : "—";
}
function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return [];
  const parseLine = line => {
    const cells = []; let cell = ""; let quoted = false;
    for (let i = 0; i < line.length; i++) {
      const char = line[i];
      if (char === '"' && line[i + 1] === '"' && quoted) { cell += '"'; i++; }
      else if (char === '"') quoted = !quoted;
      else if (char === "," && !quoted) { cells.push(cell); cell = ""; }
      else cell += char;
    }
    cells.push(cell); return cells;
  };
  const headers = parseLine(lines[0]).map(item => item.trim());
  return lines.slice(1).filter(Boolean).map(line => Object.fromEntries(parseLine(line).map((value, index) => [headers[index], value.trim()])));
}
function summarizeCsv(rawRows) {
  const groups = new Map();
  rawRows.forEach(row => {
    if (!row.accuracy || !row.macro_f1 || !row.roc_auc) return;
    const key = [row.model || "Model", row.arm || "Arm", row.fixed_strategy || ""].join("|");
    if (!groups.has(key)) groups.set(key, { values: [], model: row.model || "Model", arm: row.arm || "Arm", strategy: row.fixed_strategy || "", test_n: 3174 });
    groups.get(key).values.push(row);
  });
  return [...groups.values()].map(group => {
    const avg = field => group.values.reduce((sum, row) => sum + Number(row[field]), 0) / group.values.length;
    const armName = group.arm === "fixed" ? `Fixed · ${group.strategy || "strategy"}` : group.arm[0].toUpperCase() + group.arm.slice(1);
    return { model: group.model, arm: armName, accuracy: avg("accuracy"), macro_f1: avg("macro_f1"), roc_auc: avg("roc_auc"), test_n: group.test_n };
  });
}
function updateRouter() {
  const brightness = Number(document.querySelector("#brightness").value);
  const contrast = Number(document.querySelector("#contrast").value);
  const noise = Number(document.querySelector("#noise").value);
  document.querySelector("#brightness-value").value = brightness;
  document.querySelector("#contrast-value").value = contrast;
  document.querySelector("#noise-value").value = noise;
  const needsClahe = brightness < 70 || brightness > 190 || contrast < 25;
  const noisy = noise > 18;
  let strategy = "Standard preprocessing";
  let reason = "No configured quality threshold is crossed.";
  if (needsClahe && noisy) { strategy = "CLAHE + Wiener filtering"; reason = "Brightness / contrast and noise thresholds are crossed."; }
  else if (needsClahe) { strategy = "CLAHE"; reason = "Brightness or contrast is outside the configured range."; }
  else if (noisy) { strategy = "Wiener filtering"; reason = "Noise estimate is above the configured threshold."; }
  document.querySelector("#route-result").textContent = strategy;
  document.querySelector("#route-reason").textContent = reason;
}

renderTable(pilotRows);
renderChart(pilotRows);
document.querySelectorAll(".slider-control input").forEach(input => input.addEventListener("input", updateRouter));
updateRouter();
document.querySelector("#csv-file").addEventListener("change", event => {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    const rows = summarizeCsv(parseCsv(String(reader.result)));
    if (!rows.length) {
      document.querySelector("#import-status").textContent = "Could not find the expected comparison columns (model, arm, accuracy, macro_f1, roc_auc). The pilot values remain displayed.";
      return;
    }
    renderTable(rows); renderChart(rows);
    document.querySelector("#import-status").textContent = `Showing ${rows.length} measured model / preprocessing combinations from ${file.name}. Values are averaged across rows with matching model and strategy.`;
  };
  reader.readAsText(file);
});

const imageInput = document.querySelector("#xray-file");
const predictButton = document.querySelector("#predict-button");
let selectedImage = null;
imageInput.addEventListener("change", () => {
  selectedImage = imageInput.files[0] || null;
  document.querySelector("#prediction-result").hidden = true;
  if (!selectedImage) {
    document.querySelector("#selected-file").textContent = "No image selected";
    document.querySelector("#preview-state").textContent = "WAITING FOR IMAGE";
    predictButton.disabled = true;
    return;
  }
  document.querySelector("#selected-file").textContent = `${selectedImage.name} · ${(selectedImage.size / 1024 / 1024).toFixed(2)} MB`;
  document.querySelector("#preview-state").textContent = "IMAGE READY";
  document.querySelector("#xray-preview").src = URL.createObjectURL(selectedImage);
  document.querySelector("#xray-preview").hidden = false;
  document.querySelector(".preview-placeholder").hidden = true;
  const allowed = /\.(png|jpe?g|bmp|tiff?)$/i.test(selectedImage.name);
  predictButton.disabled = !allowed || selectedImage.size > 12 * 1024 * 1024;
  document.querySelector("#prediction-status").textContent = allowed && selectedImage.size <= 12 * 1024 * 1024
    ? "Ready. Run the model when you are ready."
    : "Choose a PNG, JPG, JPEG, BMP or TIFF image smaller than 12 MB.";
});

predictButton.addEventListener("click", async () => {
  if (!selectedImage) return;
  predictButton.disabled = true;
  predictButton.textContent = "Analyzing image…";
  document.querySelector("#prediction-status").textContent = "Running the local model. The image is not saved.";
  try {
    const bytes = new Uint8Array(await selectedImage.arrayBuffer());
    let binary = "";
    for (let offset = 0; offset < bytes.length; offset += 0x8000) {
      binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
    }
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_base64: btoa(binary) }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Prediction request failed.");
    document.querySelector("#top-label").textContent = result.top_label;
    document.querySelector("#top-score").textContent = `Model score ${(result.top_score * 100).toFixed(1)}%`;
    document.querySelector("#scores-list").innerHTML = result.scores.map(item => `<div class="score-item"><span>${escapeHtml(item.label)}</span><div><i style="width:${(item.score * 100).toFixed(1)}%"></i></div><b>${(item.score * 100).toFixed(1)}%</b></div>`).join("");
    document.querySelector("#quality-readout").textContent = `Applied ${result.preprocessing} · brightness ${result.quality.brightness.toFixed(1)} · contrast ${result.quality.contrast.toFixed(1)} · noise estimate ${result.quality.noise.toFixed(1)}`;
    const timing = result.timing_comparison;
    document.querySelector("#timing-body").innerHTML = [
      ["No enhancement", timing.raw],
      [`${result.preprocessing} path`, timing.processed],
    ].map(([name, row]) => `<tr><td>${escapeHtml(name)}</td><td>${escapeHtml(row.top_label)}</td><td>${formatMs(row.preparation_ms)}</td><td>${formatMs(row.inference_ms)}</td><td><strong>${formatMs(row.total_ms)}</strong></td></tr>`).join("");
    document.querySelector("#timing-note").textContent = `Device: ${timing.device}. ${timing.note}`;
    document.querySelector("#prediction-result").hidden = false;
    document.querySelector("#prediction-status").textContent = "Finished locally. This dataset-label estimate is not a diagnosis.";
  } catch (error) {
    document.querySelector("#prediction-status").textContent = error instanceof TypeError
      ? "Could not reach the local model. Start the demo using the command in README.md, then reload this page."
      : error.message;
  } finally {
    predictButton.disabled = !selectedImage;
    predictButton.innerHTML = 'Run research model <span>→</span>';
  }
});
