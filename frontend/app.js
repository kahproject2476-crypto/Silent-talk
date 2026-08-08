/* ── SilentTalk — frontend logic ─────────────────────────────────────────── */
'use strict';

// ── State ─────────────────────────────────────────────────────────────────
const state = {
  mode: 'sign',
  cameraRunning: false,
  stream: null,
  ws: null,
  wsReady: false,
  _wsResolve: null,       // used by send-wait pattern
  sentence: [],
  lastGesture: null,
  lastConf: 0,
  mediaRecorder: null,
  recordedChunks: [],
  recordedBlob: null,
  isRecording: false,
};

// ── DOM refs ──────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const $$ = sel => document.querySelector(sel);

const els = {
  // nav
  navTabs:       document.querySelectorAll('.nav-tab'),
  pages:         document.querySelectorAll('.page'),
  modelStatus:   $('modelStatus'),

  // sign mode
  cameraFrame:   $('cameraFrame'),
  annotatedFeed: $('annotatedFeed'),
  cameraIdle:    $('cameraIdle'),
  gestureIdle:   $('gestureIdle'),
  gestureResult: $('gestureResult'),
  gestureWord:   $('gestureWord'),
  gestureConfLabel: $('gestureConfLabel'),
  confBarFill:   $('confBarFill'),
  startCameraBtn:$('startCameraBtn'),
  stopCameraBtn: $('stopCameraBtn'),
  sentencePlaceholder: $('sentencePlaceholder'),
  sentenceText:  $('sentenceText'),
  sentenceBox:   $('sentenceBox'),
  addWordBtn:    $('addWordBtn'),
  backspaceBtn:  $('backspaceBtn'),
  clearBtn:      $('clearBtn'),
  translateBtn:  $('translateBtn'),
  translateSpinner: $('translateSpinner'),
  tuluSection:   $('tuluSection'),
  tuluText:      $('tuluText'),
  audioPlayer:   $('audioPlayer'),

  // speech mode
  speechTabs:    document.querySelectorAll('.speech-tab'),
  stabPanels:    document.querySelectorAll('.stab-panel'),
  uploadZone:    $('uploadZone'),
  audioFileInput:$('audioFileInput'),
  browseBtn:     $('browseBtn'),
  uploadPreview: $('uploadPreview'),
  fileBadge:     $('fileBadge'),
  previewPlayer: $('previewPlayer'),
  transcribeBtn: $('transcribeBtn'),
  sttSpinner:    $('sttSpinner'),
  transcriptBox: $('transcriptBox'),
  transcriptText:$('transcriptText'),
  clearTranscriptBtn: $('clearTranscriptBtn'),
  recordBtn:     $('recordBtn'),
  recordRing:    $('recordRing'),
  recordHint:    $('recordHint'),
  transcribeRecordBtn: $('transcribeRecordBtn'),

  // about
  aboutModelInfo: $('aboutModelInfo'),

  // toast
  toast: $('toast'),
};

// ── Toast ──────────────────────────────────────────────────────────────────
let _toastTimer = null;
function toast(msg, duration = 3000) {
  els.toast.textContent = msg;
  els.toast.classList.remove('hidden');
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => els.toast.classList.add('hidden'), duration);
}

// ── Mode switching ─────────────────────────────────────────────────────────
function setMode(mode) {
  state.mode = mode;
  els.navTabs.forEach(t => t.classList.toggle('active', t.dataset.mode === mode));
  els.pages.forEach(p => p.classList.toggle('active', p.id === `page-${mode}`));
  if (mode === 'about') loadModelStatus();
}

els.navTabs.forEach(tab => {
  tab.addEventListener('click', () => setMode(tab.dataset.mode));
});

// ── Load model status (navbar + about page) ────────────────────────────────
async function loadModelStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();

    // Navbar badge
    els.modelStatus.classList.toggle('ready', data.model_loaded);
    els.modelStatus.classList.toggle('error', !data.model_loaded);
    els.modelStatus.querySelector('.status-text').textContent =
      data.model_loaded ? `Model Ready · ${data.label_count} classes` : 'No Model';

    // About card
    if (data.model_loaded) {
      const chips = data.labels.map(l => `<span class="label-chip">${l}</span>`).join('');
      els.aboutModelInfo.innerHTML = `
        <div class="model-info-grid">
          <div class="model-info-row">
            <span class="label">Classifier</span>
            <span class="value">Random Forest (200 trees)</span>
          </div>
          <div class="model-info-row">
            <span class="label">Classes</span>
            <span class="value">${data.label_count} ISL gestures</span>
          </div>
          <div class="model-info-row">
            <span class="label">Confidence threshold</span>
            <span class="value">${(data.confidence_threshold * 100).toFixed(0)}%</span>
          </div>
          <div class="labels-wrap">${chips}</div>
        </div>`;
    } else {
      els.aboutModelInfo.innerHTML = `
        <p style="font-size:0.82rem;color:var(--muted)">
          No model found. Run: <code>python scripts/train_model.py</code>
        </p>`;
    }
  } catch {
    els.modelStatus.querySelector('.status-text').textContent = 'Server offline';
  }
}

loadModelStatus();

// ── WebSocket ──────────────────────────────────────────────────────────────
function connectWS() {
  return new Promise((resolve, reject) => {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${proto}://${location.host}/ws/gesture`);
    state.ws = ws;

    const timeout = setTimeout(() => reject(new Error('WS connect timeout')), 5000);

    ws.onopen = () => {
      clearTimeout(timeout);
      state.wsReady = true;
      resolve(ws);
    };
    ws.onclose = () => {
      state.wsReady = false;
      state.ws = null;
    };
    ws.onerror = (e) => {
      clearTimeout(timeout);
      state.wsReady = false;
      reject(e);
    };
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === 'result') {
        handleGestureResult(msg);
        // Signal the frame loop that server finished processing
        if (state._wsResolve) { state._wsResolve(); state._wsResolve = null; }
      }
    };
  });
}

function handleGestureResult(msg) {
  // Update annotated feed
  if (msg.annotated) {
    els.annotatedFeed.src = msg.annotated;
    els.annotatedFeed.classList.remove('hidden');
    els.cameraIdle.classList.add('hidden');
  }

  if (msg.gesture) {
    state.lastGesture = msg.gesture;
    state.lastConf    = msg.confidence;

    els.gestureIdle.classList.add('hidden');
    els.gestureResult.classList.remove('hidden');
    els.gestureWord.textContent = msg.gesture;

    const pct = Math.round(msg.confidence * 100);
    els.gestureConfLabel.textContent = `${pct}% confidence`;
    els.confBarFill.style.width = `${pct}%`;
  } else {
    els.gestureIdle.classList.remove('hidden');
    els.gestureResult.classList.add('hidden');
  }
}

// ── Camera loop ────────────────────────────────────────────────────────────
const _canvas = document.createElement('canvas');
const _ctx    = _canvas.getContext('2d');
const _video  = document.createElement('video');
_video.autoplay = true;
_video.playsInline = true;
_video.muted = true;

// Wait for video to have actual dimensions
function waitForVideo(v) {
  return new Promise(resolve => {
    if (v.readyState >= 2 && v.videoWidth > 0) { resolve(); return; }
    v.addEventListener('canplay', function handler() {
      v.removeEventListener('canplay', handler);
      resolve();
    });
  });
}

// Send one frame and wait for the server to respond before sending the next.
// This prevents flooding the WebSocket queue.
async function frameLoop() {
  while (state.cameraRunning) {
    if (!state.wsReady || !state.ws) {
      await new Promise(r => setTimeout(r, 100));
      continue;
    }

    _canvas.width  = _video.videoWidth  || 640;
    _canvas.height = _video.videoHeight || 480;
    _ctx.drawImage(_video, 0, 0);
    const b64 = _canvas.toDataURL('image/jpeg', 0.65);

    // Send frame
    state.ws.send(JSON.stringify({ type: 'frame', data: b64 }));

    // Wait for server response (max 2s) before sending next frame
    await new Promise(resolve => {
      state._wsResolve = resolve;
      setTimeout(() => { if (state._wsResolve) { state._wsResolve = null; resolve(); } }, 2000);
    });
  }
}

async function startCamera() {
  // Disable button immediately to prevent double-clicks
  els.startCameraBtn.disabled = true;

  // 1. Get webcam stream
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
      audio: false,
    });
  } catch (e) {
    toast('Camera permission denied or not available');
    els.startCameraBtn.disabled = false;
    return;
  }

  // 2. Connect WebSocket first — wait until ready
  if (!state.ws || !state.wsReady) {
    try {
      await connectWS();
    } catch (e) {
      toast('Cannot connect to server. Is python server.py running?');
      state.stream.getTracks().forEach(t => t.stop());
      state.stream = null;
      els.startCameraBtn.disabled = false;
      return;
    }
  }

  // 3. Start video
  _video.srcObject = state.stream;
  await _video.play();
  await waitForVideo(_video);

  // 4. Update UI
  state.cameraRunning = true;
  els.startCameraBtn.classList.add('hidden');
  els.startCameraBtn.disabled = false;
  els.stopCameraBtn.classList.remove('hidden');

  // 5. Start the async frame loop
  frameLoop();
}

function stopCamera() {
  state.cameraRunning = false;

  // Resolve any pending wait so the loop exits cleanly
  if (state._wsResolve) { state._wsResolve(); state._wsResolve = null; }

  if (state.stream) {
    state.stream.getTracks().forEach(t => t.stop());
    state.stream = null;
  }
  _video.srcObject = null;

  if (state.ws && state.wsReady) {
    state.ws.send(JSON.stringify({ type: 'reset' }));
  }

  // Reset UI
  els.startCameraBtn.classList.remove('hidden');
  els.stopCameraBtn.classList.add('hidden');
  els.annotatedFeed.classList.add('hidden');
  els.cameraIdle.classList.remove('hidden');
  els.gestureIdle.classList.remove('hidden');
  els.gestureResult.classList.add('hidden');
  state.lastGesture = null;
}

els.startCameraBtn.addEventListener('click', startCamera);
els.stopCameraBtn.addEventListener('click', stopCamera);

// ── Sentence builder ───────────────────────────────────────────────────────
function updateSentenceUI() {
  const text = state.sentence.join(' ');
  if (text) {
    els.sentencePlaceholder.classList.add('hidden');
    els.sentenceText.classList.remove('hidden');
    els.sentenceText.textContent = text;
    els.translateBtn.disabled = false;
  } else {
    els.sentencePlaceholder.classList.remove('hidden');
    els.sentenceText.classList.add('hidden');
    els.translateBtn.disabled = true;
  }
}

els.addWordBtn.addEventListener('click', () => {
  if (!state.lastGesture) { toast('No gesture detected yet'); return; }
  state.sentence.push(state.lastGesture);
  updateSentenceUI();
});

els.backspaceBtn.addEventListener('click', () => {
  state.sentence.pop();
  updateSentenceUI();
});

els.clearBtn.addEventListener('click', () => {
  state.sentence = [];
  els.tuluSection.classList.add('hidden');
  updateSentenceUI();
});

// ── Translate ──────────────────────────────────────────────────────────────
els.translateBtn.addEventListener('click', async () => {
  const text = state.sentence.join(' ');
  if (!text) return;

  els.translateBtn.disabled = true;
  els.translateSpinner.classList.remove('hidden');
  els.tuluSection.classList.add('hidden');

  try {
    const res = await fetch('/api/translate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();

    els.tuluText.textContent = data.tulu || data.english;
    els.tuluSection.classList.remove('hidden');

    if (data.audio_url) {
      els.audioPlayer.src = data.audio_url;
      els.audioPlayer.load();
      els.audioPlayer.play().catch(() => {});
    } else {
      els.audioPlayer.src = '';
    }
  } catch {
    toast('Translation failed — is the server running?');
  } finally {
    els.translateBtn.disabled = false;
    els.translateSpinner.classList.add('hidden');
  }
});

// ── Speech tab switcher ────────────────────────────────────────────────────
els.speechTabs.forEach(tab => {
  tab.addEventListener('click', () => {
    els.speechTabs.forEach(t => t.classList.remove('active'));
    els.stabPanels.forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    $(`stab-${tab.dataset.stab}`).classList.add('active');
  });
});

// ── File upload ────────────────────────────────────────────────────────────
els.browseBtn.addEventListener('click', () => els.audioFileInput.click());
els.uploadZone.addEventListener('click', e => {
  if (e.target === els.browseBtn) return;
  els.audioFileInput.click();
});
els.uploadZone.addEventListener('dragover', e => {
  e.preventDefault();
  els.uploadZone.classList.add('drag-over');
});
els.uploadZone.addEventListener('dragleave', () => els.uploadZone.classList.remove('drag-over'));
els.uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  els.uploadZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleAudioFile(file);
});
els.audioFileInput.addEventListener('change', () => {
  if (els.audioFileInput.files[0]) handleAudioFile(els.audioFileInput.files[0]);
});

function handleAudioFile(file) {
  els.uploadZone.classList.add('hidden');
  els.uploadPreview.classList.remove('hidden');
  els.fileBadge.textContent = `🎵 ${file.name}`;
  const url = URL.createObjectURL(file);
  els.previewPlayer.src = url;
  els.transcribeBtn._file = file;
}

async function transcribeAudio(file) {
  els.sttSpinner.classList.remove('hidden');
  els.transcriptBox.classList.add('hidden');

  const form = new FormData();
  form.append('audio', file);

  try {
    const res = await fetch('/api/transcribe', { method: 'POST', body: form });
    const data = await res.json();
    const text = data.transcript || '';

    if (text) {
      els.transcriptText.textContent = text;
      els.transcriptBox.classList.remove('hidden');
    } else {
      toast('No speech detected in the audio');
    }
  } catch {
    toast('Transcription failed — is the server running?');
  } finally {
    els.sttSpinner.classList.add('hidden');
  }
}

els.transcribeBtn.addEventListener('click', () => {
  if (els.transcribeBtn._file) transcribeAudio(els.transcribeBtn._file);
});

els.clearTranscriptBtn.addEventListener('click', () => {
  els.transcriptBox.classList.add('hidden');
  els.transcriptText.textContent = '';
  // Reset upload zone
  els.uploadZone.classList.remove('hidden');
  els.uploadPreview.classList.add('hidden');
  els.audioFileInput.value = '';
});

// ── Browser mic recording ──────────────────────────────────────────────────
els.recordBtn.addEventListener('click', async () => {
  if (!state.isRecording) {
    // Start
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      state.recordedChunks = [];
      state.mediaRecorder = new MediaRecorder(stream);
      state.mediaRecorder.ondataavailable = e => {
        if (e.data.size > 0) state.recordedChunks.push(e.data);
      };
      state.mediaRecorder.onstop = () => {
        state.recordedBlob = new Blob(state.recordedChunks, { type: 'audio/webm' });
        stream.getTracks().forEach(t => t.stop());
        els.transcribeRecordBtn.classList.remove('hidden');
        els.recordHint.textContent = 'Recording ready — click Transcribe';
      };
      state.mediaRecorder.start();
      state.isRecording = true;
      els.recordRing.classList.add('recording');
      els.recordHint.textContent = 'Recording… click to stop';
      els.transcribeRecordBtn.classList.add('hidden');
    } catch {
      toast('Microphone permission denied');
    }
  } else {
    // Stop
    state.mediaRecorder.stop();
    state.isRecording = false;
    els.recordRing.classList.remove('recording');
  }
});

els.transcribeRecordBtn.addEventListener('click', () => {
  if (state.recordedBlob) {
    const file = new File([state.recordedBlob], 'recording.webm', { type: 'audio/webm' });
    transcribeAudio(file);
    els.transcribeRecordBtn.classList.add('hidden');
    els.recordHint.textContent = 'Click to start recording';
  }
});

// ── Init ──────────────────────────────────────────────────────────────────
updateSentenceUI();
