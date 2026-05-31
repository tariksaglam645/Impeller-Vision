// ImpellerVision frontend
//  - Aydınlatma efekti: tarayıcıda CSS filtresiyle ANLIK önizleme (0 gecikme)
//  - Karar + ısı haritası: debounce'lu (~300ms) sunucu çağrısı (gerçek zamanlı his)
const $ = (id) => document.getElementById(id);

let currentFile = null;
let currentURL = null;      // canlı önizleme için objectURL
let debounceTimer = null;
let reqId = 0;              // eski yanıtları yok saymak için

const brightness = $("brightness");
const contrast = $("contrast");

// --- Sağlık göstergesi ------------------------------------------------------
fetch("/health")
  .then((r) => r.json())
  .then((d) => {
    const h = $("health");
    if (d.model_loaded) { h.classList.add("ok"); h.title = "Model hazır (" + d.device + ")"; }
    else { h.classList.add("down"); h.title = "Model yüklenmedi"; }
  })
  .catch(() => { $("health").classList.add("down"); $("health").title = "Sunucuya ulaşılamadı"; });

// --- Görüntü yükleme --------------------------------------------------------
function setImage(file) {
  currentFile = file;
  if (currentURL) URL.revokeObjectURL(currentURL);
  currentURL = URL.createObjectURL(file);
  const img = $("imgInput");
  img.src = currentURL;            // her zaman istemci orijinali (canlı filtre uygulanır)
  img.classList.add("imgInput-live");
  applyFilter();
  runPredict(false);              // ilk tahmin: tam spinner
}

const drop = $("drop");
$("file").addEventListener("change", (e) => { if (e.target.files.length) setImage(e.target.files[0]); });
["dragover", "dragenter"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("drag"); }));
drop.addEventListener("drop", (e) => { if (e.dataTransfer.files.length) setImage(e.dataTransfer.files[0]); });

// --- Örnek görüntüler -------------------------------------------------------
document.querySelectorAll(".thumb").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const url = btn.dataset.src;
    const blob = await (await fetch(url)).blob();
    setImage(new File([blob], url.split("/").pop(), { type: blob.type }));
  });
});

// --- Aydınlatma: anlık filtre + debounce'lu tahmin --------------------------
function applyFilter() {
  $("imgInput").style.filter = `brightness(${brightness.value}) contrast(${contrast.value})`;
}
function syncLabels() {
  $("bval").textContent = parseFloat(brightness.value).toFixed(2);
  $("cval").textContent = parseFloat(contrast.value).toFixed(2);
}
function schedulePredict() {
  if (!currentFile) return;
  clearTimeout(debounceTimer);
  setBusy(true);
  debounceTimer = setTimeout(() => runPredict(true), 300);
}
[brightness, contrast].forEach((el) => {
  el.addEventListener("input", () => { syncLabels(); applyFilter(); clearActivePreset(); schedulePredict(); });
});

function clearActivePreset() {
  document.querySelectorAll(".preset").forEach((b) => b.classList.remove("active"));
}
document.querySelectorAll(".preset").forEach((btn) => {
  btn.addEventListener("click", () => {
    clearActivePreset();
    btn.classList.add("active");
    brightness.value = btn.dataset.b;
    contrast.value = btn.dataset.c;
    syncLabels();
    applyFilter();
    if (currentFile) { clearTimeout(debounceTimer); runPredict(true); }
  });
});

// --- Tahmin -----------------------------------------------------------------
function setBusy(on) {
  $("updating").classList.toggle("hidden", !on);
  $("imgHeat").parentElement.classList.toggle("busy", on);
}

async function runPredict(quiet) {
  if (!currentFile) return;
  const myId = ++reqId;
  if (!quiet) {
    $("placeholder").classList.add("hidden");
    $("result").classList.add("hidden");
    $("spinner").classList.remove("hidden");
  } else {
    setBusy(true);
  }

  const fd = new FormData();
  fd.append("file", currentFile);
  fd.append("brightness", brightness.value);
  fd.append("contrast", contrast.value);

  try {
    const res = await fetch("/predict", { method: "POST", body: fd });
    if (!res.ok) throw new Error(await res.text());
    const d = await res.json();
    if (myId !== reqId) return;        // daha yeni bir istek var → bu yanıtı yok say
    render(d);
  } catch (err) {
    if (myId !== reqId) return;
    $("spinner").classList.add("hidden");
    $("placeholder").classList.remove("hidden");
    $("placeholder").innerHTML = '<span class="ph-icon">⚠</span><p>Hata: ' + err.message + "</p>";
  } finally {
    if (myId === reqId) setBusy(false);
  }
}

function render(d) {
  $("spinner").classList.add("hidden");
  $("placeholder").classList.add("hidden");
  $("result").classList.remove("hidden");

  const pass = d.decision === "PASS";
  const badge = $("badge");
  badge.textContent = pass ? "✓ PASS" : "✗ FAIL";
  badge.className = "badge " + (pass ? "pass" : "fail");

  $("conftext").textContent = "Güven (" + d.label + ")";
  $("confpct").textContent = d.confidence + "%";
  const fill = $("conffill");
  fill.className = "conf-fill " + (pass ? "pass" : "fail");
  fill.style.width = d.confidence + "%";

  $("defprob").textContent = d.defect_prob + "%";
  $("okprob").textContent = d.ok_prob + "%";

  $("bboxnote").classList.toggle("hidden", !d.bbox);
  // imgInput istemci orijinali + canlı CSS filtresi (anlık) — sunucu kopyasıyla değiştirmeyiz
  $("imgHeat").src = d.heatmap;
}

syncLabels();

// Ziyaret bildirimi — oturum başına bir kez, sessiz (sayfayı etkilemez)
if (!sessionStorage.getItem("iv_tracked")) {
  sessionStorage.setItem("iv_tracked", "1");
  fetch("/track", { method: "POST" }).catch(() => {});
}

// Paylaşılabilir otomatik demo: .../#demo ilk kusurlu örneği otomatik çalıştırır
if (location.hash === "#demo") {
  window.addEventListener("load", () => {
    const t = document.querySelector('.thumb[data-src*="defect_1"]');
    if (t) t.click();
  });
}
