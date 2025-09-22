// ATC MVP Frontend Logic

const LM_SEQUENCE = [
  { key: 'withers', label: 'Withers (highest point over shoulders)' },
  { key: 'brisket', label: 'Brisket (bottom of chest between forelimbs)' },
  { key: 'front_hoof', label: 'Front hoof (ground contact point)' },
  { key: 'shoulder', label: 'Shoulder (point of shoulder)' },
  { key: 'hook', label: 'Hook (hip bone)' },
  { key: 'pin', label: 'Pin (pin bone / tailhead)' },
];

const els = {
  species: document.getElementById('species'),
  imageInput: document.getElementById('imageInput'),
  autoDetect: document.getElementById('autoDetect'),
  autoSpecies: document.getElementById('autoSpecies'),
  startLM: document.getElementById('startLM'),
  undoLM: document.getElementById('undoLM'),
  clearLM: document.getElementById('clearLM'),
  lmList: document.getElementById('lmList'),
  lmStatus: document.getElementById('lmStatus'),
  analyzeBtn: document.getElementById('analyzeBtn'),
  exportLink: document.getElementById('exportLink'),
  refreshRecords: document.getElementById('refreshRecords'),
  recordsList: document.getElementById('recordsList'),
  canvas: document.getElementById('canvas'),
  results: document.getElementById('results'),
};

const ctx = els.canvas.getContext('2d');

const state = {
  imgEl: null,
  imgUrl: null,
  imgNaturalWidth: 0,
  imgNaturalHeight: 0,
  drawScale: 1,
  drawDx: 0,
  drawDy: 0,
  placing: false,
  activeIndex: -1,
  placed: [], // {name, x, y} in image pixel coordinates
};

function resetCanvas() {
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
  ctx.fillStyle = '#f8fafc';
  ctx.fillRect(0, 0, els.canvas.width, els.canvas.height);
}

function fitAndDrawImage() {
  resetCanvas();
  if (!state.imgEl) return;

  const cW = els.canvas.width;
  const cH = els.canvas.height;
  const iW = state.imgNaturalWidth;
  const iH = state.imgNaturalHeight;
  const scale = Math.min(cW / iW, cH / iH);
  const dW = iW * scale;
  const dH = iH * scale;
  const dx = (cW - dW) / 2;
  const dy = (cH - dH) / 2;

  state.drawScale = scale;
  state.drawDx = dx;
  state.drawDy = dy;

  ctx.drawImage(state.imgEl, dx, dy, dW, dH);
  drawOverlay();
}

function toImageCoords(cx, cy) {
  const x = (cx - state.drawDx) / state.drawScale;
  const y = (cy - state.drawDy) / state.drawScale;
  return { x, y };
}

function toCanvasCoords(ix, iy) {
  const x = ix * state.drawScale + state.drawDx;
  const y = iy * state.drawScale + state.drawDy;
  return { x, y };
}

function drawOverlay() {
  // draw placed points and helper lines
  if (!state.imgEl) return;

  // Semi-transparent overlay for clarity
  // ctx.fillStyle = 'rgba(255,255,255,0.0)';
  // ctx.fillRect(0, 0, els.canvas.width, els.canvas.height);

  // draw points
  const ptColor = '#0f766e';
  ctx.lineWidth = 2;

  // Helper lines from key relationships if present
  const getPoint = (name) => state.placed.find((p) => p.name === name);
  const shoulder = getPoint('shoulder');
  const pin = getPoint('pin');
  const withers = getPoint('withers');
  const brisket = getPoint('brisket');
  const frontHoof = getPoint('front_hoof');
  const hook = getPoint('hook');

  // body length line: shoulder -> pin
  if (shoulder && pin) {
    const a = toCanvasCoords(shoulder.x, shoulder.y);
    const b = toCanvasCoords(pin.x, pin.y);
    ctx.strokeStyle = '#1f2937';
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  // withers to front hoof vertical (height)
  if (withers && frontHoof) {
    const a = toCanvasCoords(withers.x, withers.y);
    const b = toCanvasCoords(frontHoof.x, frontHoof.y);
    ctx.strokeStyle = '#6b7280';
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // withers to brisket vertical (chest depth)
  if (withers && brisket) {
    const a = toCanvasCoords(withers.x, withers.y);
    const b = toCanvasCoords(brisket.x, brisket.y);
    ctx.strokeStyle = '#0ea5e9';
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  // hook to pin line (rump angle)
  if (hook && pin) {
    const a = toCanvasCoords(hook.x, hook.y);
    const b = toCanvasCoords(pin.x, pin.y);
    ctx.strokeStyle = '#ef4444';
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  // draw points with labels
  state.placed.forEach((p, idx) => {
    const c = toCanvasCoords(p.x, p.y);
    ctx.fillStyle = ptColor;
    ctx.beginPath();
    ctx.arc(c.x, c.y, 4, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#111827';
    ctx.font = '12px system-ui, sans-serif';
    ctx.fillText(`${p.name}`, c.x + 6, c.y - 6);
  });
}

function updateLmList() {
  els.lmList.innerHTML = '';
  LM_SEQUENCE.forEach((step, idx) => {
    const li = document.createElement('li');
    const placed = state.placed[idx];
    const isActive = idx === state.activeIndex;
    li.textContent = `${idx + 1}. ${step.label}` + (placed ? ' ✓' : isActive ? ' (next)' : '');
    if (isActive) li.style.fontWeight = '600';
    els.lmList.appendChild(li);
  });
}

function updateLmStatus() {
  if (!state.placing) {
    els.lmStatus.textContent = 'Click "Start Landmarking" to begin';
    return;
  }
  if (state.activeIndex >= 0 && state.activeIndex < LM_SEQUENCE.length) {
    els.lmStatus.textContent = `Place: ${LM_SEQUENCE[state.activeIndex].label}`;
  } else {
    els.lmStatus.textContent = 'All landmarks placed';
  }
}

function startLandmarking() {
  if (!state.imgEl) {
    alert('Please upload an image first.');
    return;
  }
  state.placing = true;
  state.activeIndex = 0;
  state.placed = [];
  fitAndDrawImage();
  updateLmList();
  updateLmStatus();
}

function undoLandmark() {
  if (!state.placing || state.placed.length === 0) return;
  state.placed.pop();
  state.activeIndex = Math.max(0, state.placed.length);
  fitAndDrawImage();
  updateLmList();
  updateLmStatus();
}

function clearLandmarks() {
  state.placed = [];
  state.activeIndex = state.placing ? 0 : -1;
  fitAndDrawImage();
  updateLmList();
  updateLmStatus();
}

function onCanvasClick(ev) {
  if (!state.placing) return;
  if (els.autoDetect.checked) return; // skip if auto
  if (state.activeIndex < 0 || state.activeIndex >= LM_SEQUENCE.length) return;

  const rect = els.canvas.getBoundingClientRect();
  const cx = ev.clientX - rect.left;
  const cy = ev.clientY - rect.top;

  // ensure click lies within drawn image area
  if (
    cx < state.drawDx ||
    cy < state.drawDy ||
    cx > state.drawDx + state.imgNaturalWidth * state.drawScale ||
    cy > state.drawDy + state.imgNaturalHeight * state.drawScale
  ) {
    return;
  }

  const { x, y } = toImageCoords(cx, cy);
  const step = LM_SEQUENCE[state.activeIndex];
  state.placed.push({ name: step.key, x, y });

  state.activeIndex = state.placed.length;
  if (state.activeIndex >= LM_SEQUENCE.length) {
    state.activeIndex = LM_SEQUENCE.length; // done
  }

  fitAndDrawImage();
  updateLmList();
  updateLmStatus();
}

async function fetchRecords() {
  try {
    const resp = await fetch('/api/records?limit=10');
    const data = await resp.json();
    renderRecords(data);
  } catch (e) {
    console.error('Failed to load records', e);
  }
}

function renderRecords(records) {
  els.recordsList.innerHTML = '';
  records.reverse().forEach((rec) => {
    const li = document.createElement('li');

    const meta = document.createElement('div');
    meta.className = 'meta';
    const ts = new Date(rec.timestamp || '').toLocaleString();
    const det = rec.detected_species ? ` (detected: ${rec.detected_species})` : '';
    meta.textContent = `${rec.id} • ${rec.species}${det} • ${ts} • Overall: ${rec.overall_score ?? '—'}`;

    const actions = document.createElement('div');
    actions.className = 'actions';

    const sendBtn = document.createElement('button');
    sendBtn.className = 'secondary';
    sendBtn.textContent = 'Send to BPA (stub)';
    sendBtn.onclick = async () => {
      try {
        sendBtn.disabled = true;
        const r = await fetch('/api/send_to_bpa', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ record_id: rec.id }),
        });
        const out = await r.json();
        alert(`Sent: ${out.sent}\nStatus: ${out.status_code ?? ''}\nInfo: ${out.reason ?? out.error ?? ''}`);
      } catch (e) {
        alert('Failed to send to BPA');
      } finally {
        sendBtn.disabled = false;
      }
    };

    actions.appendChild(sendBtn);

    li.appendChild(meta);
    li.appendChild(actions);
    els.recordsList.appendChild(li);
  });
}

function showResults(data) {
  els.results.classList.remove('hidden');
  const { features, scores, overall_score, record_id, species, detected_species, species_confidence, species_method } = data || {};

  const traitLines = [
    `body_length_ratio: ${features?.body_length_ratio?.toFixed?.(3) ?? features?.body_length_ratio ?? '—'} (score ${scores?.body_length_ratio ?? '—'})`,
    `chest_depth_ratio: ${features?.chest_depth_ratio?.toFixed?.(3) ?? features?.chest_depth_ratio ?? '—'} (score ${scores?.chest_depth_ratio ?? '—'})`,
    `rump_angle_deg: ${features?.rump_angle_deg?.toFixed?.(1) ?? features?.rump_angle_deg ?? '—'} (score ${scores?.rump_angle_deg ?? '—'})`,
  ].join('\n');

  els.results.innerHTML = `
    <h3>Results</h3>
    <div><strong>Record ID:</strong> ${record_id ?? '—'}</div>
    <div><strong>Species used:</strong> ${species ?? '—'}</div>
    <div><strong>Detected species:</strong> ${detected_species ?? '—'} ${species_confidence != null ? `(conf ${Math.round(species_confidence * 100)}%)` : ''} ${species_method ? `[${species_method}]` : ''}</div>
    <div><strong>Overall Score:</strong> ${overall_score ?? '—'}</div>
    <pre>${traitLines}</pre>
  `;
}

async function analyze() {
  const files = els.imageInput.files;
  if (!files || files.length === 0) {
    alert('Please select an image.');
    return;
  }
  const auto = !!els.autoDetect.checked;
  const needAllLandmarks = !auto;
  if (needAllLandmarks && state.placed.length < LM_SEQUENCE.length) {
    alert('Please place all guided landmarks or enable Auto silhouette.');
    return;
  }

  const fd = new FormData();
  fd.append('image', files[0]);
  fd.append('species', els.species.value || 'cattle');
  fd.append('auto_detect', String(auto));
  fd.append('auto_species', String(!!els.autoSpecies?.checked));
  fd.append('save_record', 'true');
  if (!auto && state.placed.length) {
    fd.append('landmarks_json', JSON.stringify(state.placed));
  }

  try {
    els.analyzeBtn.disabled = true;
    const resp = await fetch('/api/score', { method: 'POST', body: fd });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err?.detail || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    showResults(data);
    await fetchRecords();
  } catch (e) {
    alert(`Failed: ${e.message || e}`);
  } finally {
    els.analyzeBtn.disabled = false;
  }
}

function loadImageToCanvas(file) {
  if (state.imgUrl) URL.revokeObjectURL(state.imgUrl);
  state.imgUrl = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    state.imgEl = img;
    state.imgNaturalWidth = img.naturalWidth;
    state.imgNaturalHeight = img.naturalHeight;
    state.placed = [];
    state.activeIndex = -1;
    state.placing = false;
    fitAndDrawImage();
    updateLmList();
    updateLmStatus();
  };
  img.src = state.imgUrl;
}

// Event bindings
els.imageInput.addEventListener('change', () => {
  const files = els.imageInput.files;
  if (files && files[0]) {
    loadImageToCanvas(files[0]);
  }
});

els.startLM.addEventListener('click', startLandmarking);
els.undoLM.addEventListener('click', undoLandmark);
els.clearLM.addEventListener('click', clearLandmarks);
els.canvas.addEventListener('click', onCanvasClick);
els.analyzeBtn.addEventListener('click', analyze);
els.refreshRecords.addEventListener('click', fetchRecords);
els.autoDetect.addEventListener('change', () => {
  if (els.autoDetect.checked) {
    // auto mode ignores landmarks
    els.lmStatus.textContent = 'Auto mode enabled — landmarking disabled';
  } else {
    updateLmStatus();
  }
});

// Initial UI
resetCanvas();
updateLmList();
updateLmStatus();
fetchRecords();
