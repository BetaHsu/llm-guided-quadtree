// Co-Authored / Unauthored — an LLM-guided generative quadtree.
//
// A quadtree runs continuously: it picks a live space, cuts it into four, and
// elsewhere merges four siblings back into their parent. The balance between the
// two drifts over time, so the canvas never settles. Two things can intervene:
// a click from the viewer, and a language model asked to re-tune how future
// splits are shaped.
//
// The model's reach is deliberately narrow. It sets the character of new cuts
// and, optionally, which region they concentrate in. It never touches the
// split/merge rhythm, and every value it returns is clamped before use.

// Local development talks to server.js; production talks to the Vercel function.
const AI_BACKEND_URL =
  (typeof location !== 'undefined' && location.hostname && location.hostname !== 'localhost' && location.hostname !== '127.0.0.1')
    ? '/api/ai-decide'
    : 'http://localhost:8787/ai-decide';

// System defaults. Only the three keys named in AI_TUNABLE_RANGES can ever be
// overridden by the model; everything else here is fixed.
const PARAMS = {
  MAX_SPLIT_COUNT: 10,
  MIN_SPACE_SIZE: 30,
  SPLIT_MARGIN_RATIO: 0.15,

  SPLIT_WEIGHT_START: 8,
  SPLIT_WEIGHT_END: 5,
  MERGE_WEIGHT_START: 1,
  MERGE_WEIGHT_END: 3,
  WEIGHT_TRANSITION_SPLITS: 150,

  MERGE_DEPTH_BIAS: 1,

  SPLITS_PER_FRAME: 1,
  FRAME_RATE: 6,

  BG_COLOR: [30, 30, 30],
  LINE_COLOR: [200, 200, 200],
  LINE_WEIGHT: 1.5,

  CANVAS_SIZE: 640,
  PANEL_WIDTH: 340,
  PANEL_TEXT_COLOR: [200, 200, 200],
  PANEL_TEXT_SIZE: 14,

  MERGE_FLASH_DURATION: 300,
  MERGE_FLASH_COLOR: [255, 255, 255],

  USER_ACTION_LOG_DURATION: 800,

  RESET_WEIGHTS_ON_FULL_MERGE: true,
};

// The model's entire surface area: three shape parameters, each with a declared
// safe range. `desc` is sent to the model so it knows which way each value pushes.
// Anything outside this table is unreachable from a model response.
const AI_TUNABLE_RANGES = {
  MAX_SPLIT_COUNT:    { min: 8,    max: 30,   desc: 'How many times one space can be subdivided. Higher = deeper, finer fragments are allowed to grow' },
  MIN_SPACE_SIZE:     { min: 3,    max: 30,   desc: 'A fragment below this edge length (px) stops splitting. Smaller = far tinier fragments are allowed; larger = the canvas settles at a coarser, blockier state' },
  SPLIT_MARGIN_RATIO: { min: 0.02, max: 0.4,  desc: 'How far the cut point can stray from the centre. Smaller = the four pieces differ wildly in size and long strips appear; larger (towards 0.5) = the four pieces come out close in size, square and even' },
};

// Optional regional bias. A named region multiplies the weight of split candidates
// whose centre falls inside it. A bias, not a lock: other areas keep evolving, and
// merging is unaffected.
const AI_REGION_WEIGHT_MULTIPLIER = 6;
const AI_REGIONS = ['top-left', 'top', 'top-right', 'left', 'center', 'right', 'bottom-left', 'bottom', 'bottom-right'];

// Live quadtree state. Nodes are stored flat in a Map; childIds === null means leaf.
let nodes = new Map();
let nextId = 0;
let totalSplitsSoFar = 0;
let totalMergesSoFar = 0;

let weightTransitionStartAt = 0;

let activeFlashes = [];

let lastUserActionLog = null;

function logUserAction(text) {
  lastUserActionLog = { text, startTime: millis() };
}

// aiMode: null = rule-based, 'thinking' = request in flight, 'active' = a model
// decision is applied. aiOverride holds that decision until it is dropped.
let aiMode = null;
let aiOverride = null;

// Model overrides shadow PARAMS for the three tunable keys only. Every other
// parameter always reads PARAMS, whatever the model returned.
function getEffectiveParams() {
  if (aiMode !== 'active' || !aiOverride) return PARAMS;
  return { ...PARAMS, ...aiOverride.params };
}

function getEffectiveWeights() {
  return getCurrentWeights();
}

const REGION_COLS = ['left', 'center', 'right'];
const REGION_ROWS = ['top', 'middle', 'bottom'];
function regionNameAt(x, y) {
  const col = REGION_COLS[constrain(floor((x / PARAMS.CANVAS_SIZE) * 3), 0, 2)];
  const row = REGION_ROWS[constrain(floor((y / PARAMS.CANVAS_SIZE) * 3), 0, 2)];
  if (row === 'middle' && col === 'center') return 'center';
  if (row === 'middle') return col;
  if (col === 'center') return row;
  return `${row}-${col}`;
}

function nodeInRegion(node, region) {
  return regionNameAt(node.x + node.w / 2, node.y + node.h / 2) === region;
}

function setup() {
  const canvas = createCanvas(PARAMS.CANVAS_SIZE + PARAMS.PANEL_WIDTH, PARAMS.CANVAS_SIZE);

  canvas.elt.oncontextmenu = (e) => e.preventDefault();
  frameRate(PARAMS.FRAME_RATE);
  resetScene();
}
function draw() {
  processPendingClick();
  for (let i = 0; i < PARAMS.SPLITS_PER_FRAME; i++) {
    stepOnce();
  }
  redrawAll();
}

function stepOnce() {
  const { splitWeight, mergeWeight } = getEffectiveWeights();
  const totalWeight = splitWeight + mergeWeight;
  const roll = random(totalWeight);
  if (roll < splitWeight) {
    splitOnce();
  } else {
    tryMergeOnce();
  }
}

// Split and merge weights drift one way from _START to _END and then lock.
// Rhythm is not reachable by the model.
function getCurrentWeights() {
  const progress = totalSplitsSoFar - weightTransitionStartAt;
  const t = constrain(progress / PARAMS.WEIGHT_TRANSITION_SPLITS, 0, 1);
  return {
    splitWeight: lerp(PARAMS.SPLIT_WEIGHT_START, PARAMS.SPLIT_WEIGHT_END, t),
    mergeWeight: lerp(PARAMS.MERGE_WEIGHT_START, PARAMS.MERGE_WEIGHT_END, t),
  };
}

function splitOnce() {
  const p = getEffectiveParams();
  const splittable = Array.from(nodes.values())
    .filter(n => n.childIds === null && !n.isTiny && n.splitCount < p.MAX_SPLIT_COUNT);
  if (splittable.length === 0) return;
  const picked = pickWithRegionTendency(splittable);
  doSplit(picked);
}

// Applies the model's regional bias when one is active; plain weighted random otherwise.
function pickWithRegionTendency(candidates, baseWeightFn = () => 1) {
  const regionActive = aiMode === 'active' && aiOverride && aiOverride.region;
  if (!regionActive) {
    return candidates[weightedIndex(candidates.map(baseWeightFn))];
  }
  const weights = candidates.map(n =>
    baseWeightFn(n) * (nodeInRegion(n, aiOverride.region) ? AI_REGION_WEIGHT_MULTIPLIER : 1)
  );
  return candidates[weightedIndex(weights)];
}

function weightedIndex(weights) {
  const totalWeight = weights.reduce((sum, w) => sum + w, 0);
  let roll = random(totalWeight);
  for (let i = 0; i < weights.length; i++) {
    roll -= weights[i];
    if (roll < 0) return i;
  }
  return weights.length - 1;
}

function doSplit(space) {
  const p = getEffectiveParams();

  const margin = min(space.w, space.h) * p.SPLIT_MARGIN_RATIO;
  const splitX = space.x + random(margin, space.w - margin);
  const splitY = space.y + random(margin, space.h - margin);
  space.splitX = splitX;
  space.splitY = splitY;
  totalSplitsSoFar++;
  const nextSplitCount = space.splitCount + 1;

  const leftW = splitX - space.x;
  const rightW = space.x + space.w - splitX;
  const topH = splitY - space.y;
  const bottomH = space.y + space.h - splitY;
  const candidateRects = [
    { x: space.x, y: space.y, w: leftW,  h: topH    },
    { x: splitX,  y: space.y, w: rightW, h: topH    },
    { x: space.x, y: splitY,  w: leftW,  h: bottomH },
    { x: splitX,  y: splitY,  w: rightW, h: bottomH },
  ];

  const childIds = [];
  for (const rect of candidateRects) {
    const isTiny = rect.w < p.MIN_SPACE_SIZE || rect.h < p.MIN_SPACE_SIZE;
    const child = {
      id: nextId++,
      x: rect.x, y: rect.y, w: rect.w, h: rect.h,
      splitCount: nextSplitCount,
      parentId: space.id,
      childIds: null,
      isTiny,
    };
    nodes.set(child.id, child);
    childIds.push(child.id);
  }
  space.childIds = childIds;
}

function tryMergeOnce() {
  const mergeCandidates = Array.from(nodes.values()).filter(n => n.childIds !== null);
  if (mergeCandidates.length === 0) return;
  const weights = mergeCandidates.map(depthWeight);
  const picked = mergeCandidates[weightedIndex(weights)];
  doMerge(picked);
}

function depthWeight(node) {
  return Math.pow(node.splitCount, PARAMS.MERGE_DEPTH_BIAS);
}

function doMerge(target) {
  activeFlashes.push({ x: target.x, y: target.y, w: target.w, h: target.h, startTime: millis() });

  collectDescendantsToMerge(target);
  target.childIds = null;
  target.isTiny = false;
  totalMergesSoFar++;

  maybeResetWeightTransition();
}

// If the canvas merges all the way back to empty, restart the drift so the raised
// merge weight doesn't immediately collapse it again.
function maybeResetWeightTransition() {
  if (!PARAMS.RESET_WEIGHTS_ON_FULL_MERGE) return;
  if (nodes.size === 1) {
    weightTransitionStartAt = totalSplitsSoFar;
  }
}

function collectDescendantsToMerge(node) {
  for (const childId of node.childIds) {
    const child = nodes.get(childId);
    if (child.childIds !== null) {
      activeFlashes.push({ x: child.x, y: child.y, w: child.w, h: child.h, startTime: millis() });
      collectDescendantsToMerge(child);
      totalMergesSoFar++;
    }
    nodes.delete(childId);
  }
}

function redrawAll() {
  background(PARAMS.BG_COLOR);
  stroke(PARAMS.LINE_COLOR[0], PARAMS.LINE_COLOR[1], PARAMS.LINE_COLOR[2]);
  strokeWeight(PARAMS.LINE_WEIGHT);
  for (const node of nodes.values()) {
    if (node.childIds === null) continue;
    line(node.splitX, node.y, node.splitX, node.y + node.h);
    line(node.x, node.splitY, node.x + node.w, node.splitY);
  }
  updateFlashes();
  drawDebugPanel();
}

function updateFlashes() {
  noStroke();
  for (let i = activeFlashes.length - 1; i >= 0; i--) {
    const flash = activeFlashes[i];
    const elapsed = millis() - flash.startTime;
    const t = constrain(elapsed / PARAMS.MERGE_FLASH_DURATION, 0, 1);

    if (t >= 1) {
      activeFlashes.splice(i, 1);
      continue;
    }

    const alpha = lerp(255, 0, t);
    fill(PARAMS.MERGE_FLASH_COLOR[0], PARAMS.MERGE_FLASH_COLOR[1], PARAMS.MERGE_FLASH_COLOR[2], alpha);
    rect(flash.x, flash.y, flash.w, flash.h);
  }
}

const PANEL_HEADING_COLOR = [140, 200, 255];

function textMixedColor(str, x, y, highlightWords, baseColor) {

  const ranges = [];
  for (const word of highlightWords) {
    let searchFrom = 0;
    while (true) {
      const idx = str.indexOf(word, searchFrom);
      if (idx === -1) break;
      ranges.push({ start: idx, end: idx + word.length });
      searchFrom = idx + word.length;
    }
  }
  ranges.sort((a, b) => a.start - b.start);

  let cursor = 0;
  let cursorX = x;
  for (const range of ranges) {
    if (range.start < cursor) continue;
    if (range.start > cursor) {
      const plain = str.slice(cursor, range.start);
      fill(baseColor[0], baseColor[1], baseColor[2]);
      text(plain, cursorX, y);
      cursorX += textWidth(plain);
    }
    const highlighted = str.slice(range.start, range.end);
    fill(PANEL_HEADING_COLOR[0], PANEL_HEADING_COLOR[1], PANEL_HEADING_COLOR[2]);
    text(highlighted, cursorX, y);
    cursorX += textWidth(highlighted);
    cursor = range.end;
  }
  if (cursor < str.length) {
    const plain = str.slice(cursor);
    fill(baseColor[0], baseColor[1], baseColor[2]);
    text(plain, cursorX, y);
  }
}

// The panel is the explanation surface, not a debug readout: current mode, what the
// model changed in plain language, and the reason it gave.
function drawDebugPanel() {
  const aliveCount = countLiveLeaves();
  const panelX = PARAMS.CANVAS_SIZE + 20;
  let lineY = 30;
  const lineHeight = 24;
  const barWidth = PARAMS.PANEL_WIDTH - 60;
  noStroke();

  textSize(PARAMS.PANEL_TEXT_SIZE);
  textAlign(LEFT, TOP);

  fill(PARAMS.PANEL_TEXT_COLOR[0], PARAMS.PANEL_TEXT_COLOR[1], PARAMS.PANEL_TEXT_COLOR[2]);
  text('Alive spaces', panelX, lineY);
  lineY += lineHeight * 0.9;
  drawAliveSpacesBar(panelX, lineY, barWidth, 14, aliveCount);
  lineY += 14 + lineHeight * 1.4;

  textMixedColor('Left Click to Split, Right Click to Merge', panelX, lineY, ['Left Click', 'Right Click'], PARAMS.PANEL_TEXT_COLOR);
  lineY += lineHeight * 0.85;
  textMixedColor('Press Space to switch mode', panelX, lineY, ['Space'], PARAMS.PANEL_TEXT_COLOR);
  lineY += lineHeight * 1.2;

  const modeLabel = aiMode === 'active' && aiOverride ? 'AI-guided' : 'Rule-based';
  textMixedColor(`Current mode: ${modeLabel}`, panelX, lineY, [modeLabel], PARAMS.PANEL_TEXT_COLOR);
  lineY += lineHeight * 1.3;

  lineY = drawAIPanelSection(panelX, lineY, lineHeight);

  if (lastUserActionLog !== null) {
    const elapsed = millis() - lastUserActionLog.startTime;
    if (elapsed < PARAMS.USER_ACTION_LOG_DURATION) {
      fill(PARAMS.PANEL_TEXT_COLOR[0], PARAMS.PANEL_TEXT_COLOR[1], PARAMS.PANEL_TEXT_COLOR[2]);
      lineY += lineHeight * 0.5;
      text(lastUserActionLog.text, panelX, lineY);
    }
  }
}

const ALIVE_SPACES_REFERENCE_MAX = 400;
function drawAliveSpacesBar(x, y, w, h, count) {
  const linearRatio = constrain(count / ALIVE_SPACES_REFERENCE_MAX, 0, 1);
  const portion = Math.sqrt(linearRatio) * w;
  noStroke();
  fill(110, 110, 110);
  rect(x, y, w, h);
  fill(215, 215, 215);
  rect(x, y, portion, h);
  fill(PARAMS.PANEL_TEXT_COLOR[0], PARAMS.PANEL_TEXT_COLOR[1], PARAMS.PANEL_TEXT_COLOR[2]);
  text(`${count}`, x + w + 8, y - 1);
}

// Internal parameter names are shown to the viewer in plain language, so the panel
// stays readable to someone who has never seen this file.
const PARAM_DISPLAY_NAMES = {
  MAX_SPLIT_COUNT: 'fragment depth',
  MIN_SPACE_SIZE: 'minimum piece size',
  SPLIT_MARGIN_RATIO: 'cut evenness',
};

function drawAIPanelSection(panelX, startY, lineHeight) {
  let lineY = startY;
  fill(PARAMS.PANEL_TEXT_COLOR[0], PARAMS.PANEL_TEXT_COLOR[1], PARAMS.PANEL_TEXT_COLOR[2]);

  if (aiMode === 'thinking') {
    text('Asking the AI what it sees...', panelX, lineY);
    return lineY + lineHeight;
  }

  if (aiMode !== 'active' || !aiOverride) {
    if (lastAIError) {
      fill(255, 130, 130);
      text('Last call failed:', panelX, lineY);
      lineY += lineHeight * 0.85;
      const wrapped = wrapText(lastAIError, 44);
      text(wrapped, panelX, lineY);

      lineY += lineHeight * (wrapped.split('\n').length * 0.85 + 0.3);
    }
    return lineY;
  }

  text('What it changed:', panelX, lineY);
  lineY += lineHeight * 0.85;
  for (const key of Object.keys(AI_TUNABLE_RANGES)) {
    const label = PARAM_DISPLAY_NAMES[key] || key;
    text(`  ${label}: ${PARAMS[key]} → ${aiOverride.params[key].toFixed(2)}`, panelX, lineY);
    lineY += lineHeight * 0.85;
  }
  text(
    aiOverride.region ? `  focusing new cuts on: ${aiOverride.region}` : '  spreading cuts evenly across canvas',
    panelX, lineY
  );
  lineY += lineHeight;
  text('What it saw and why:', panelX, lineY);
  lineY += lineHeight * 0.85;
  const wrappedReason = wrapText(aiOverride.reason, 44);
  text(wrappedReason, panelX, lineY);

  lineY += lineHeight * (wrappedReason.split('\n').length * 0.85 + 0.3);
  return lineY;
}

function wrapText(str, maxCharsPerLine) {
  const words = str.split(' ');
  const lines = [];
  let current = '';
  for (const word of words) {
    const candidate = current ? `${current} ${word}` : word;
    if (candidate.length > maxCharsPerLine && current) {
      lines.push(current);
      current = word;
    } else {
      current = candidate;
    }
  }
  if (current) lines.push(current);
  return lines.join('\n');
}

function countLiveLeaves() {
  let count = 0;
  for (const node of nodes.values()) {
    if (node.childIds === null) count++;
  }
  return count;
}

function findLeafAt(px, py) {
  if (px < 0 || px >= PARAMS.CANVAS_SIZE || py < 0 || py >= PARAMS.CANVAS_SIZE) return null;
  for (const node of nodes.values()) {
    if (node.childIds !== null) continue;
    if (px >= node.x && px < node.x + node.w && py >= node.y && py < node.y + node.h) {
      return node;
    }
  }
  return null;
}

function contextMenuEvent(e) {
  e.preventDefault();
}

// Space asks the model for a decision; space again drops it and returns to defaults.
// Ignored while a request is still in flight.
function keyPressed() {
  if (key !== ' ') return;

  if (aiMode === 'active') {
    aiMode = null;
    aiOverride = null;
    return;
  }
  if (aiMode === 'thinking') return;

  requestAIDecision();
}

// Builds the request: the current leaf list as structured data, the tunable ranges
// with their descriptions, and the response contract. The model receives numbers and
// positions, never an image. The call goes through the backend so the API key never
// reaches the browser.
async function requestAIDecision() {
  aiMode = 'thinking';

  const p = getEffectiveParams();
  const leaves = Array.from(nodes.values())
    .filter(n => n.childIds === null)
    .map(n => ({
      x: Math.round(n.x), y: Math.round(n.y),
      w: Math.round(n.w), h: Math.round(n.h),
      splitCount: n.splitCount,
      region: regionNameAt(n.x + n.w / 2, n.y + n.h / 2),
    }));

  const rangesText = Object.entries(AI_TUNABLE_RANGES)
    .map(([key, r]) => `  - ${key} (current: ${p[key]}, allowed range: ${r.min}–${r.max}): ${r.desc}`)
    .join('\n');

  const prompt = `You are controlling a generative quadtree animation. The canvas is ${PARAMS.CANVAS_SIZE}x${PARAMS.CANVAS_SIZE}px, divided into leaf spaces that continuously split into 4 smaller ones or merge back into their parent. The pace of splitting vs merging, and which leaf gets merged, is fixed and not yours to set. You only control two things: the SHAPE that new splits produce (applies everywhere on the canvas, uniformly), and optionally, WHERE splitting concentrates.

Current state: ${leaves.length} live leaves.
Full leaf list (x, y, w, h, splitCount, rough region) — this is internal data for your own analysis, not vocabulary for your explanation:
${JSON.stringify(leaves)}

Shape parameters you can set (current value → allowed range). These three interact — here are four reference combinations and the "character" they produce, to give you a wider palette than just one direction. All four are equally valid, striking choices — none is a fallback or a lesser option:
${rangesText}
  - Fine slivers: very small SPLIT_MARGIN_RATIO + small MIN_SPACE_SIZE + high MAX_SPLIT_COUNT → thin, needle-like fragments.
  - Even squares: large SPLIT_MARGIN_RATIO (near 0.4) + moderate MIN_SPACE_SIZE → grid-like, regular, orderly blocks.
  - Big open chunks: low MAX_SPLIT_COUNT + large MIN_SPACE_SIZE → the canvas stays mostly a few large open spaces, splitting is rare but each one is a big visible event.
  - Lopsided splits: very small SPLIT_MARGIN_RATIO + large MIN_SPACE_SIZE → each split produces one large piece and several tiny ones that can't fragment further — an asymmetric, "bitten-off-corner" look.
These are reference points, not the only options — you can also aim for something in between. Pick whichever character contrasts most with the canvas's current state. Across repeated calls, distribute your choices roughly evenly across all four characters — don't let "fine slivers" become a default. As a rough target: over any 10 calls, aim for something like fine slivers ~3-4 times, even squares ~3-4 times, and the other two characters splitting the remainder — but let the actual canvas state override this whenever a different character is the more striking contrast.

You should usually make splitting concentrate in one specific region rather than spreading evenly — a clear regional focus is almost always the more visually interesting, noticeable choice, and is what you should pick most of the time. Set "region" to one of: ${AI_REGIONS.join(', ')}. This only affects where NEW splits are more likely to happen — it's a bias, not a lock, other areas still keep evolving too, and it does not affect merging at all. Only set "region" to null in the rare case where the canvas is already so balanced that no single area stands out as the more interesting place to push — this should be uncommon, not your default.

Your job: look at the leaf data above (sizes, positions, how fragmented different areas already are) and choose a shape + optional region focus that will make the canvas evolve in a visually distinct, noticeably different way from how it's been running.

Then explain your choice in ONE short sentence (under 20 words) that a viewer with no knowledge of the code could understand just by looking at the canvas, and could later confirm by watching what happens next. Structure: state one visual observation about the current canvas, then state where you're pushing NEW SPLITS toward. Do not mention merging — you don't control it and its effect isn't visually obvious. Do not make claims about what other areas of the canvas will do or "stay" like — you only control where splitting concentrates, nothing else is under your control. Describe things in plain visual terms only. Do not use internal terms like "splitCount", exact pixel coordinates, exact counts, or measurements. Do not use poetic/narrative language (no metaphors, no words like "decay", "crumbling", "town").
  Good example (no region focus): "The canvas is mostly big open blocks right now, so I'm pushing new splits toward thin, needle-like fragments."
  Good example (with region focus): "The top-right is already the busiest, fine-grained part of the canvas, so I'm concentrating new splits there to push it further."
  Bad example (mentions merge, which isn't controllable): "Aggressively merging the fragmented bottom back into larger blocks while splitting the top."
  Bad example (claims control over untouched areas): "Pushing the top into thin strips while the rest stays balanced."
  Bad example (uses code jargon): "splitCount is uneven across regions, so I'm normalizing MERGE_DEPTH_BIAS."
  Bad example (too poetic): "the canvas feels like a crumbling frontier town, so I will let it decay."

Respond with ONLY a JSON object, no other text, in exactly this shape:
{"params": {"MAX_SPLIT_COUNT": number, "MIN_SPACE_SIZE": number, "SPLIT_MARGIN_RATIO": number}, "region": "one of the region names above, or null", "reason": "one short plain sentence, under 20 words"}`;

  try {
    const res = await fetch(AI_BACKEND_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt }),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    const parsed = parseAIResponse(data.text);
    applyAIDecision(parsed);
  } catch (err) {
    console.error('AI request failed:', err);
    aiMode = null;
    lastAIError = String(err.message || err);
  }
}

// Models occasionally wrap JSON in prose or a code fence. Taking the outermost braces
// is more reliable than demanding perfectly clean output.
function parseAIResponse(text) {
  const start = text.indexOf('{');
  const end = text.lastIndexOf('}');
  if (start === -1 || end === -1) throw new Error('No JSON found in the model response');
  return JSON.parse(text.slice(start, end + 1));
}

// The only place a model response reaches the system. Numbers are clamped into their
// declared ranges; region is whitelist-checked and falls back to no bias. A nonsensical
// response cannot put the system into an invalid state.
function applyAIDecision(parsed) {
  const clampedParams = {};
  for (const key of Object.keys(AI_TUNABLE_RANGES)) {
    const r = AI_TUNABLE_RANGES[key];
    const raw = Number(parsed.params?.[key]);
    clampedParams[key] = constrain(isNaN(raw) ? PARAMS[key] : raw, r.min, r.max);
  }

  const region = AI_REGIONS.includes(parsed.region) ? parsed.region : null;

  aiOverride = {
    params: clampedParams,
    region,
    reason: String(parsed.reason || '(no reason given)'),
    appliedAtSplit: totalSplitsSoFar,
  };
  aiMode = 'active';
  lastAIError = null;
}

let lastAIError = null;

let pendingClick = null;

function mousePressed() {
  pendingClick = { x: mouseX, y: mouseY, isLeftClick: mouseButton === LEFT };
}

function processPendingClick() {
  if (pendingClick === null) return;
  const { x, y, isLeftClick } = pendingClick;
  pendingClick = null;

  const leaf = findLeafAt(x, y);
  if (leaf === null) return;

  handleClickFull(leaf, isLeftClick);
}

// Left click splits, right click merges one level up. Illegal moves are ignored
// rather than redirected to some other node.
function handleClickFull(leaf, isLeftClick) {
  if (isLeftClick) {
    const p = getEffectiveParams();
    const canSplit = !leaf.isTiny && leaf.splitCount < p.MAX_SPLIT_COUNT;
    if (canSplit) {
      doSplit(leaf);
      logUserAction('User split ✓');
    }
  } else {
    if (leaf.parentId === null) return;
    const parent = nodes.get(leaf.parentId);
    doMerge(parent);
    logUserAction('User merge ✓');
  }
}

// Full reset: one uncut root node, counters zeroed, any model decision dropped.
function resetScene() {
  nextId = 0;
  totalSplitsSoFar = 0;
  totalMergesSoFar = 0;
  weightTransitionStartAt = 0;
  activeFlashes = [];
  pendingClick = null;
  lastUserActionLog = null;
  aiMode = null;
  aiOverride = null;
  lastAIError = null;
  const root = {
    id: nextId++,
    x: 0, y: 0, w: PARAMS.CANVAS_SIZE, h: PARAMS.CANVAS_SIZE,
    splitCount: 0,
    parentId: null,
    childIds: null,
    isTiny: false,
  };
  nodes = new Map([[root.id, root]]);
  redrawAll();
}