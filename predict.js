/* ============================================================
 * Pitchprice — Phase 1 · predict.js
 * Loads model.json, renders sliders, runs vanilla-JS inference.
 * (ML.js is linked from the CDN in index.html per the project
 *  spec; for a plain linear regression a direct dot-product is
 *  equivalent and requires no re-training inside the browser.)
 * ============================================================ */

'use strict';

// -- Feature → tab mapping --------------------------------------------------
const FEATURE_GROUPS = {
  physical: ['height', 'weight', 'age'],
  mental:   ['reactions', 'composure', 'vision', 'aggression', 'att_position', 'interceptions'],
  attack:   ['shot_power', 'finishing', 'long_shots', 'curve', 'fk_acc', 'penalties', 'volleys', 'heading'],
  skill:    ['ball_control', 'dribbling', 'short_pass', 'long_pass', 'crossing'],
  movement: ['acceleration', 'sprint_speed', 'agility', 'balance', 'stamina', 'strength', 'jumping'],
  defense:  ['slide_tackle', 'stand_tackle'],
  gk:       ['gk_positioning', 'gk_diving', 'gk_handling', 'gk_kicking', 'gk_reflexes']
};

// -- Human-readable labels --------------------------------------------------
const LABELS = {
  height: 'Height (cm)', weight: 'Weight (kg)', age: 'Age',
  ball_control: 'Ball control', dribbling: 'Dribbling',
  slide_tackle: 'Slide tackle', stand_tackle: 'Standing tackle',
  aggression: 'Aggression', reactions: 'Reactions',
  att_position: 'Attacking position', interceptions: 'Interceptions',
  vision: 'Vision', composure: 'Composure',
  crossing: 'Crossing', short_pass: 'Short passing', long_pass: 'Long passing',
  acceleration: 'Acceleration', stamina: 'Stamina', strength: 'Strength',
  balance: 'Balance', sprint_speed: 'Sprint speed', agility: 'Agility',
  jumping: 'Jumping', heading: 'Heading',
  shot_power: 'Shot power', finishing: 'Finishing', long_shots: 'Long shots',
  curve: 'Curve', fk_acc: 'Free-kick accuracy', penalties: 'Penalties',
  volleys: 'Volleys',
  gk_positioning: 'GK positioning', gk_diving: 'GK diving',
  gk_handling: 'GK handling',   gk_kicking: 'GK kicking',
  gk_reflexes: 'GK reflexes'
};

// -- Preset scenarios -------------------------------------------------------
// Each preset is a partial feature override (missing keys stay at defaults).
const PRESETS = {
  superstar: { // Elite striker, late-20s prime
    age: 27, height: 184, weight: 79,
    ball_control: 94, dribbling: 92, finishing: 95, shot_power: 93,
    long_shots: 88, curve: 86, fk_acc: 80, penalties: 88, volleys: 87,
    heading: 85, short_pass: 85, long_pass: 78, crossing: 78, vision: 86,
    reactions: 95, composure: 93, att_position: 95, acceleration: 90,
    sprint_speed: 91, agility: 89, balance: 84, stamina: 82, strength: 78,
    jumping: 83, aggression: 65, interceptions: 28,
    slide_tackle: 20, stand_tackle: 25,
    gk_positioning: 10, gk_diving: 10, gk_handling: 10, gk_kicking: 10, gk_reflexes: 10
  },
  midfielder: {
    age: 26, height: 178, weight: 73,
    ball_control: 91, dribbling: 87, short_pass: 93, long_pass: 88, vision: 92,
    reactions: 91, composure: 92, crossing: 82, curve: 84, fk_acc: 82,
    finishing: 78, shot_power: 82, long_shots: 85, penalties: 78, volleys: 76,
    heading: 68, att_position: 78, acceleration: 80, sprint_speed: 78,
    agility: 85, balance: 85, stamina: 88, strength: 72, jumping: 72,
    aggression: 72, interceptions: 78, slide_tackle: 68, stand_tackle: 72,
    gk_positioning: 10, gk_diving: 10, gk_handling: 10, gk_kicking: 10, gk_reflexes: 10
  },
  defender: {
    age: 28, height: 188, weight: 84,
    slide_tackle: 90, stand_tackle: 92, heading: 88, strength: 86,
    jumping: 84, aggression: 85, interceptions: 90, reactions: 86,
    composure: 84, ball_control: 70, dribbling: 62, short_pass: 76,
    long_pass: 78, crossing: 58, vision: 68, finishing: 45, shot_power: 72,
    long_shots: 55, curve: 45, fk_acc: 40, penalties: 50, volleys: 45,
    att_position: 45, acceleration: 70, sprint_speed: 74, agility: 66,
    balance: 76, stamina: 82,
    gk_positioning: 10, gk_diving: 10, gk_handling: 10, gk_kicking: 10, gk_reflexes: 10
  },
  gk: {
    age: 28, height: 192, weight: 85,
    gk_positioning: 90, gk_diving: 88, gk_handling: 89, gk_kicking: 82, gk_reflexes: 91,
    reactions: 88, composure: 86, jumping: 78, strength: 80, aggression: 40,
    ball_control: 30, dribbling: 18, short_pass: 55, long_pass: 60, crossing: 15,
    vision: 55, finishing: 15, shot_power: 55, long_shots: 18, curve: 15,
    fk_acc: 15, penalties: 30, volleys: 18, heading: 20, att_position: 15,
    acceleration: 50, sprint_speed: 50, agility: 65, balance: 58, stamina: 45,
    slide_tackle: 13, stand_tackle: 14, interceptions: 20
  },
  rookie: {
    age: 18, height: 176, weight: 70,
    ball_control: 70, dribbling: 72, short_pass: 66, long_pass: 58,
    crossing: 60, vision: 60, reactions: 70, composure: 58,
    finishing: 62, shot_power: 64, long_shots: 55, curve: 55,
    fk_acc: 50, penalties: 55, volleys: 50, heading: 55,
    att_position: 68, acceleration: 82, sprint_speed: 80, agility: 78,
    balance: 70, stamina: 72, strength: 60, jumping: 68, aggression: 55,
    interceptions: 40, slide_tackle: 40, stand_tackle: 42,
    gk_positioning: 10, gk_diving: 10, gk_handling: 10, gk_kicking: 10, gk_reflexes: 10
  }
};

// -- State -------------------------------------------------------------------
let MODEL = null;
let currentValues = {};   // feature_name -> current slider value
let activeTab = 'physical';
let liveUpdate = true;

// -- DOM refs ----------------------------------------------------------------
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const els = {
  status:       $('#status'),
  statusText:   $('.status-text'),
  sliderGrid:   $('#slider-grid'),
  predValue:    $('#prediction-value'),
  predMeta:     $('#prediction-meta'),
  predBtn:      $('#predict-btn'),
  resetBtn:     $('#reset-btn'),
  liveToggle:   $('#live-toggle'),
  infoAlgo:     $('#info-algo'),
  infoMae:      $('#info-mae'),
  infoRmse:     $('#info-rmse'),
  infoR2:       $('#info-r2')
};

// -- Init --------------------------------------------------------------------
async function init() {
  try {
    const resp = await fetch('model.json');
    if (!resp.ok) throw new Error(`HTTP ${resp.status} — is model.json in the same folder?`);
    MODEL = await resp.json();

    // initialise current values from defaults
    MODEL.feature_names.forEach((name, i) => {
      currentValues[name] = MODEL.feature_defaults[i];
    });

    buildSliders();
    showTab('physical');
    populateModelInfo();
    predict();

    els.status.classList.add('ready');
    els.statusText.textContent = `Model loaded · ${MODEL.feature_names.length} features`;
  } catch (err) {
    console.error(err);
    els.status.classList.add('error');
    els.statusText.textContent = `Failed to load model: ${err.message}`;
  }

  // Wire controls
  els.predBtn.addEventListener('click', () => predict(true));
  els.resetBtn.addEventListener('click', resetSliders);
  els.liveToggle.addEventListener('change', (e) => { liveUpdate = e.target.checked; });

  $$('.tab').forEach((btn) => {
    btn.addEventListener('click', () => showTab(btn.dataset.tab));
  });
  $$('.chip').forEach((btn) => {
    btn.addEventListener('click', () => applyPreset(btn.dataset.preset));
  });
}

// -- Slider generation -------------------------------------------------------
function buildSliders() {
  els.sliderGrid.innerHTML = '';
  MODEL.feature_names.forEach((name, i) => {
    const range = MODEL.feature_ranges[name];
    const defaultVal = MODEL.feature_defaults[i];
    const group = findGroup(name);

    const wrap = document.createElement('div');
    wrap.className = 'slider-item';
    wrap.dataset.feature = name;
    wrap.dataset.group = group;

    // round bounds sensibly
    const minV = Math.floor(range.min);
    const maxV = Math.ceil(range.max);
    const step = Number.isInteger(defaultVal) ? 1 : 0.5;

    wrap.innerHTML = `
      <div class="slider-head">
        <span class="slider-name">${LABELS[name] || name}</span>
        <span class="slider-val" data-val="${name}">${formatVal(defaultVal)}</span>
      </div>
      <input type="range"
             min="${minV}" max="${maxV}" step="${step}"
             value="${defaultVal}"
             data-input="${name}"
             aria-label="${LABELS[name] || name}" />
      <div class="slider-range">
        <span>${minV}</span><span>${maxV}</span>
      </div>
    `;
    els.sliderGrid.appendChild(wrap);

    const input = wrap.querySelector('input[type="range"]');
    const valueSpan = wrap.querySelector('[data-val]');

    input.addEventListener('input', (e) => {
      const v = parseFloat(e.target.value);
      currentValues[name] = v;
      valueSpan.textContent = formatVal(v);
      if (liveUpdate) predictDebounced();
    });
  });
}

function findGroup(featureName) {
  for (const [grp, feats] of Object.entries(FEATURE_GROUPS)) {
    if (feats.includes(featureName)) return grp;
  }
  return 'physical';
}

function formatVal(v) {
  if (Number.isInteger(v)) return v.toString();
  return v.toFixed(1);
}

// -- Tab switching -----------------------------------------------------------
function showTab(tab) {
  activeTab = tab;
  $$('.tab').forEach((b) => b.classList.toggle('active', b.dataset.tab === tab));
  $$('.slider-item').forEach((el) => {
    el.classList.toggle('visible', el.dataset.group === tab);
  });
}

// -- Preset application ------------------------------------------------------
function applyPreset(name) {
  const preset = PRESETS[name];
  if (!preset) return;

  // Start from defaults, then overlay preset
  MODEL.feature_names.forEach((feat, i) => {
    const val = preset[feat] !== undefined ? preset[feat] : MODEL.feature_defaults[i];
    currentValues[feat] = val;
    const input = document.querySelector(`input[data-input="${feat}"]`);
    const valSpan = document.querySelector(`[data-val="${feat}"]`);
    if (input)   input.value = val;
    if (valSpan) valSpan.textContent = formatVal(val);
  });
  predict(true);
}

// -- Reset -------------------------------------------------------------------
function resetSliders() {
  MODEL.feature_names.forEach((feat, i) => {
    const def = MODEL.feature_defaults[i];
    currentValues[feat] = def;
    const input = document.querySelector(`input[data-input="${feat}"]`);
    const valSpan = document.querySelector(`[data-val="${feat}"]`);
    if (input)   input.value = def;
    if (valSpan) valSpan.textContent = formatVal(def);
  });
  predict(true);
}

// -- Prediction --------------------------------------------------------------
function predict(animate = false) {
  if (!MODEL) return;

  // 1. Assemble raw feature vector in the exact training order
  const raw = MODEL.feature_names.map((f) => currentValues[f]);

  // 2. Standard-scale (reproduces sklearn's StandardScaler client-side)
  const scaled = raw.map((x, i) => (x - MODEL.feature_means[i]) / MODEL.feature_stds[i]);

  // 3. Linear combination: y = x · w + b   (in log-space)
  let yLog = MODEL.intercept;
  for (let i = 0; i < scaled.length; i++) {
    yLog += scaled[i] * MODEL.coef[i];
  }

  // 4. Invert the log1p transform used during training
  let yEur = Math.expm1(yLog);
  if (yEur < 0) yEur = 0;            // prevent negative prices for weak rookies
  if (!Number.isFinite(yEur)) yEur = 0;

  // 5. Render
  if (animate) {
    els.predValue.classList.remove('updating');
    void els.predValue.offsetWidth;   // restart animation
    els.predValue.classList.add('updating');
  }
  els.predValue.textContent = formatEuro(yEur);

  // Meta info: context about the prediction
  els.predMeta.innerHTML = bucketMessage(yEur);
}

// Debounce live updates for smoother slider feel
let debounceTimer = null;
function predictDebounced() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => predict(false), 40);
}

// -- Helpers -----------------------------------------------------------------
function formatEuro(v) {
  if (v >= 1_000_000) {
    return `€\u00A0${(v / 1_000_000).toLocaleString('en-US', { maximumFractionDigits: 2 })}M`;
  }
  if (v >= 1_000) {
    return `€\u00A0${(v / 1_000).toLocaleString('en-US', { maximumFractionDigits: 0 })}K`;
  }
  return `€\u00A0${Math.round(v).toLocaleString('en-US')}`;
}

function bucketMessage(v) {
  if (v >= 80_000_000) return 'Generational talent — competes for the Ballon d\'Or.';
  if (v >= 30_000_000) return 'World-class — Champions League starter territory.';
  if (v >= 10_000_000) return 'Top-tier first-team player for a European club.';
  if (v >= 3_000_000)  return 'Solid starter at a mid-table club or strong rotation piece.';
  if (v >= 1_000_000)  return 'Squad player / promising prospect.';
  if (v >= 300_000)    return 'Lower-division regular or emerging academy graduate.';
  return 'Youth / developmental contract.';
}

function populateModelInfo() {
  const m = MODEL.metrics;
  els.infoAlgo.textContent = MODEL.model_type || 'LinearRegression';
  els.infoMae.textContent  = `€ ${formatCompact(m.MAE_eur)}`;
  els.infoRmse.textContent = `€ ${formatCompact(m.RMSE_eur)}`;
  els.infoR2.textContent   = m.R2_log.toFixed(3);
}

function formatCompact(n) {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000)     return `${(n / 1_000).toFixed(0)}K`;
  return n.toFixed(0);
}

// ------------- boot ---------------------------------------------------------
document.addEventListener('DOMContentLoaded', init);
