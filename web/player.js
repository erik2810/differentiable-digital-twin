// Differentiable Digital Twin -- trajectory player.
//
// Loads an optimised trajectory exported by `ddtwin optimise` (scene.json + positions.bin)
// and renders the deforming structure with a WebGPU renderer (WebGL2 fallback). Node
// displacements are small by design, so the player exaggerates them by a user-controlled
// factor -- exactly how an FEA post-processor visualises deflection.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const $ = (id) => document.getElementById(id);
const DATA_DIR = "data";

// ------------------------------------------------------------------ colour map
// density 0 (soft / void) -> deep blue ; density 1 (stiff / material) -> amber.
const STOPS = [
  [0.0, [0.11, 0.22, 0.44]],
  [0.4, [0.23, 0.51, 0.96]],
  [0.72, [0.74, 0.46, 0.12]],
  [1.0, [0.96, 0.62, 0.04]],
];
function colormap(d) {
  d = Math.min(1, Math.max(0, d));
  for (let i = 1; i < STOPS.length; i++) {
    if (d <= STOPS[i][0]) {
      const [t0, c0] = STOPS[i - 1];
      const [t1, c1] = STOPS[i];
      const u = (d - t0) / (t1 - t0 || 1);
      return c0.map((c, k) => c + (c1[k] - c) * u);
    }
  }
  return STOPS[STOPS.length - 1][1];
}

// ------------------------------------------------------------------ state
const state = {
  scene: null, // parsed scene.json
  rest: null, // Float32Array (N*3) render-space flat panel, centred
  density: null, // Float32Array (N)
  positions: null, // Float32Array (frames*N*3) raw data-space trajectory
  restData: null, // Float32Array (N*3) data-space rest
  frames: 0,
  nodes: 0,
  center: [0, 0], // (cx, cz) data-space centroid
  scale: 20,
  relief: 0.8,
  frame: 0,
  playing: true,
  speed: 1,
};

// data-space (x, y, z) -> render-space (X=x, Y=z, Z=y): the in-plane deflection (z) becomes
// screen-vertical so the cantilever visibly bends "down" under the load.
function buildRenderRest() {
  const N = state.nodes;
  const rd = state.restData;
  let cx = 0;
  let cz = 0;
  for (let i = 0; i < N; i++) {
    cx += rd[i * 3 + 0];
    cz += rd[i * 3 + 2];
  }
  cx /= N;
  cz /= N;
  state.center = [cx, cz];
  const r = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    r[i * 3 + 0] = rd[i * 3 + 0] - cx; // X = x
    r[i * 3 + 1] = rd[i * 3 + 2] - cz; // Y = z
    r[i * 3 + 2] = 0; // Z (depth) = 0, relief added at render time
  }
  state.rest = r;
}

// ------------------------------------------------------------------ three.js
let renderer, scene, camera, controls, mesh, edges, supports, loadArrow, geom;

async function initThree() {
  const canvas = $("scene");
  let hasWebGPU = false;
  try {
    hasWebGPU = !!navigator.gpu && !!(await navigator.gpu.requestAdapter());
  } catch {
    hasWebGPU = false;
  }
  renderer = new THREE.WebGPURenderer({ canvas, antialias: true, forceWebGL: !hasWebGPU });
  await renderer.init();
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  resize();

  const label = hasWebGPU ? "WebGPU" : "WebGL2 (fallback)";
  $("gpu-label").textContent = label;
  $("gpu-badge").title = `Rendering backend: ${label}`;

  scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x05070d, 28, 64);

  camera = new THREE.PerspectiveCamera(42, 1, 0.1, 200);
  camera.position.set(-7, 8, 26);

  controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 8;
  controls.maxDistance = 60;

  // Lights matched to the blue + amber palette.
  scene.add(new THREE.AmbientLight(0x33415f, 1.4));
  const key = new THREE.DirectionalLight(0xdfe9ff, 2.4);
  key.position.set(-8, 16, 14);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0x3b82f6, 1.6);
  rim.position.set(10, 4, -12);
  scene.add(rim);
  const fill = new THREE.PointLight(0xf59e0b, 0.7, 80);
  fill.position.set(12, -6, 10);
  scene.add(fill);

  // Faint reference grid under the structure.
  const grid = new THREE.GridHelper(48, 24, 0x1b2742, 0x121a30);
  grid.position.y = -9;
  scene.add(grid);

  window.addEventListener("resize", resize);
}

function resize() {
  const main = $("scene").parentElement;
  const w = main.clientWidth;
  const h = main.clientHeight;
  renderer.setSize(w, h, false);
  if (camera) {
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
}

// ------------------------------------------------------------------ build mesh
function buildMesh() {
  const N = state.nodes;
  const faces = state.scene.topology.faces;

  geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(N * 3), 3));
  geom.setIndex(faces);

  const colors = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    const c = colormap(state.density[i]);
    colors[i * 3] = c[0];
    colors[i * 3 + 1] = c[1];
    colors[i * 3 + 2] = c[2];
  }
  geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));

  const mat = new THREE.MeshStandardMaterial({
    vertexColors: true,
    metalness: 0.25,
    roughness: 0.55,
    side: THREE.DoubleSide,
    emissive: new THREE.Color(0x0a1326),
    emissiveIntensity: 0.6,
    flatShading: false,
  });
  mesh = new THREE.Mesh(geom, mat);
  scene.add(mesh);

  // Spring topology overlay (the truss) -- hidden until "wireframe" is toggled.
  const egeom = new THREE.BufferGeometry();
  egeom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(N * 3), 3));
  egeom.setIndex(state.scene.topology.edges);
  edges = new THREE.LineSegments(
    egeom,
    new THREE.LineBasicMaterial({ color: 0x6f9bff, transparent: true, opacity: 0.35 }),
  );
  edges.visible = false;
  scene.add(edges);

  buildSupports();
  buildLoadArrow();
  updateGeometry(0);
}

function buildSupports() {
  const grp = new THREE.Group();
  const pinned = state.scene.boundary.pinned;
  const mat = new THREE.MeshStandardMaterial({ color: 0x94a3b8, metalness: 0.6, roughness: 0.4 });
  for (let i = 0; i < state.nodes; i++) {
    if (!pinned[i]) continue;
    const cube = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.5, 1.1), mat);
    cube.position.set(state.rest[i * 3], state.rest[i * 3 + 1], 0);
    grp.add(cube);
  }
  supports = grp;
  scene.add(grp);
}

function buildLoadArrow() {
  // Load is in data -z -> render -Y (downward). Anchor at the tip-edge centre.
  const N = state.nodes;
  const loaded = state.scene.boundary.loaded;
  let x = 0;
  let y = 0;
  let n = 0;
  for (let i = 0; i < N; i++) {
    if (!loaded[i]) continue;
    x += state.rest[i * 3];
    y += state.rest[i * 3 + 1];
    n++;
  }
  const origin = new THREE.Vector3(x / n, y / n + 6, 0);
  loadArrow = new THREE.ArrowHelper(
    new THREE.Vector3(0, -1, 0),
    origin,
    5.5,
    0xf59e0b,
    1.8,
    1.1,
  );
  scene.add(loadArrow);
}

// Write vertex positions for a (fractional) frame: rest + scale*disp + relief.
function updateGeometry(frameF) {
  const N = state.nodes;
  const F = state.frames;
  const f0 = Math.floor(frameF) % F;
  const f1 = (f0 + 1) % F;
  const u = frameF - Math.floor(frameF);
  const p = state.positions;
  const rd = state.restData;
  const [cx, cz] = state.center;

  const pos = geom.attributes.position.array;
  const epos = edges.geometry.attributes.position.array;
  for (let i = 0; i < N; i++) {
    const b0 = f0 * N * 3 + i * 3;
    const b1 = f1 * N * 3 + i * 3;
    // interpolate the data-space position between the two frames
    const dx = (p[b0] + (p[b1] - p[b0]) * u) - rd[i * 3];
    const dy = (p[b0 + 1] + (p[b1 + 1] - p[b0 + 1]) * u) - rd[i * 3 + 1];
    const dz = (p[b0 + 2] + (p[b1 + 2] - p[b0 + 2]) * u) - rd[i * 3 + 2];
    const X = rd[i * 3] - cx + state.scale * dx;
    const Y = rd[i * 3 + 2] - cz + state.scale * dz;
    const Z = state.density[i] * state.relief + state.scale * dy;
    pos[i * 3] = X;
    pos[i * 3 + 1] = Y;
    pos[i * 3 + 2] = Z;
    epos[i * 3] = X;
    epos[i * 3 + 1] = Y;
    epos[i * 3 + 2] = Z;
  }
  geom.attributes.position.needsUpdate = true;
  geom.computeVertexNormals();
  edges.geometry.attributes.position.needsUpdate = true;
}

// ------------------------------------------------------------------ data load
async function loadData() {
  const sceneRes = await fetch(`${DATA_DIR}/scene.json`, { cache: "no-store" });
  if (!sceneRes.ok) throw new Error(`scene.json -> HTTP ${sceneRes.status}`);
  const sc = await sceneRes.json();
  const { nodes, frames } = sc.meta;

  const binRes = await fetch(`${DATA_DIR}/positions.bin`, { cache: "no-store" });
  if (!binRes.ok) throw new Error(`positions.bin -> HTTP ${binRes.status}`);
  const buf = await binRes.arrayBuffer();
  const positions = new Float32Array(buf);
  if (positions.length !== frames * nodes * 3) {
    throw new Error(`positions.bin size ${positions.length} != ${frames}x${nodes}x3`);
  }

  state.scene = sc;
  state.nodes = nodes;
  state.frames = frames;
  state.density = Float32Array.from(sc.design.density);
  state.positions = positions;
  state.restData = Float32Array.from(sc.topology.rest);
  buildRenderRest();
}

// ------------------------------------------------------------------ HUD
function fillHud() {
  const sc = state.scene;
  const d = sc.design;
  // Headline: stiffness ratio = uniform compliance / optimised compliance (higher = stiffer).
  const ratio = d.compliance > 0 ? d.uniform_compliance / d.compliance : 1;
  $("reduction").textContent = ratio.toFixed(1);
  $("subnote").textContent =
    `compliance ${d.uniform_compliance.toFixed(3)} → ${d.compliance.toFixed(3)}` +
    `  ·  tip deflection −${Math.round(d.deflection_improvement * 100)}%`;

  const maxc = Math.max(d.uniform_compliance, d.compliance) || 1;
  $("bar-uniform").style.width = `${(d.uniform_compliance / maxc) * 100}%`;
  $("bar-optimised").style.width = `${(d.compliance / maxc) * 100}%`;
  $("val-uniform").textContent = d.uniform_compliance.toFixed(3);
  $("val-optimised").textContent = d.compliance.toFixed(3);

  $("kv-mesh").textContent = `${sc.meta.rows} × ${sc.meta.cols}`;
  $("kv-dofs").textContent = `${sc.meta.nodes} / ${sc.topology.edges.length / 2}`;
  $("kv-budget").textContent = `${Math.round(d.volume_fraction * 100)} %`;
  $("kv-defl").textContent = d.tip_deflection.toFixed(4);
  $("kv-sims").textContent = sc.history.loss.length;
  $("kv-loss").textContent = d.compliance.toFixed(4);

  $("subtitle").textContent =
    `cantilever ${sc.meta.rows}×${sc.meta.cols} · ${sc.meta.frames} frames`;
  $("frame-tot").textContent = state.frames - 1;
  $("seek").max = state.frames - 1;

  drawSpark(sc.history.loss);
}

function drawSpark(loss) {
  const cv = $("spark");
  const ctx = cv.getContext("2d");
  const W = cv.width;
  const H = cv.height;
  ctx.clearRect(0, 0, W, H);
  if (!loss || loss.length < 2) return;
  const lo = Math.min(...loss);
  const hi = Math.max(...loss);
  const x = (i) => (i / (loss.length - 1)) * (W - 4) + 2;
  const y = (v) => H - 4 - ((v - lo) / (hi - lo || 1)) * (H - 8);

  // area fill
  ctx.beginPath();
  ctx.moveTo(x(0), H);
  loss.forEach((v, i) => ctx.lineTo(x(i), y(v)));
  ctx.lineTo(x(loss.length - 1), H);
  ctx.closePath();
  const grad = ctx.createLinearGradient(0, 0, 0, H);
  grad.addColorStop(0, "rgba(59,130,246,0.35)");
  grad.addColorStop(1, "rgba(59,130,246,0.02)");
  ctx.fillStyle = grad;
  ctx.fill();

  // line
  ctx.beginPath();
  loss.forEach((v, i) => (i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))));
  ctx.strokeStyle = "#3b82f6";
  ctx.lineWidth = 1.6;
  ctx.stroke();

  // endpoint
  ctx.beginPath();
  ctx.arc(x(loss.length - 1), y(loss[loss.length - 1]), 2.6, 0, 7);
  ctx.fillStyle = "#f59e0b";
  ctx.fill();
}

// ------------------------------------------------------------------ controls
function wireControls() {
  const playIcon = $("play-icon");
  const setPlaying = (p) => {
    state.playing = p;
    playIcon.innerHTML = p
      ? '<path d="M6 5h4v14H6zM14 5h4v14h-4z" />'
      : '<path d="M8 5v14l11-7z" />';
  };
  $("play").addEventListener("click", () => setPlaying(!state.playing));

  $("seek").addEventListener("input", (e) => {
    state.frame = Number(e.target.value);
    setPlaying(false);
    updateGeometry(state.frame);
    $("frame-cur").textContent = Math.floor(state.frame);
  });

  document.querySelectorAll(".speed button").forEach((b) => {
    b.addEventListener("click", () => {
      state.speed = Number(b.dataset.speed);
      document.querySelectorAll(".speed button").forEach((o) =>
        o.setAttribute("aria-pressed", String(o === b)),
      );
    });
  });

  $("scale").addEventListener("input", (e) => {
    state.scale = Number(e.target.value);
    $("scale-val").innerHTML = `${state.scale}&times;`;
    updateGeometry(state.frame);
  });
  $("relief").addEventListener("input", (e) => {
    state.relief = Number(e.target.value);
    $("relief-val").textContent = state.relief.toFixed(1);
    updateGeometry(state.frame);
  });

  const toggle = (id, fn, init) => {
    const el = $(id);
    el.setAttribute("aria-pressed", String(init));
    fn(init);
    el.addEventListener("click", () => {
      const next = el.getAttribute("aria-pressed") !== "true";
      el.setAttribute("aria-pressed", String(next));
      fn(next);
    });
  };
  toggle("t-wire", (v) => (edges.visible = v), false);
  toggle("t-supports", (v) => (supports.visible = v), true);
  toggle("t-load", (v) => (loadArrow.visible = v), true);
  toggle("t-spin", (v) => (controls.autoRotate = v), false);
  controls.autoRotateSpeed = 0.8;
}

// ------------------------------------------------------------------ loop
let last = performance.now();
function loop() {
  const now = performance.now();
  const dt = (now - last) / 1000;
  last = now;

  if (state.playing) {
    state.frame = (state.frame + dt * 24 * state.speed) % state.frames;
    updateGeometry(state.frame);
    $("seek").value = state.frame;
    $("frame-cur").textContent = Math.floor(state.frame);
  }
  controls.update();
  renderer.render(scene, camera);
}

// ------------------------------------------------------------------ boot
function fail(msg) {
  $("spin").style.display = "none";
  $("loader-msg").className = "err";
  $("loader-msg").innerHTML =
    `<b>Could not load trajectory.</b><br/>${msg}<br/><br/>` +
    "Generate it and serve over HTTP:<br/>" +
    "<code>uv run ddtwin optimise --float32</code><br/>" +
    "<code>cd web && python -m http.server 8080</code><br/>" +
    "then open <code>http://localhost:8080</code>";
}

async function main() {
  try {
    await initThree();
    await loadData();
    buildMesh();
    fillHud();
    wireControls();
    document.querySelectorAll(".speed button").forEach((b) =>
      b.setAttribute("aria-pressed", String(Number(b.dataset.speed) === 1)),
    );
    $("loader").classList.add("hidden");
    renderer.setAnimationLoop(loop);
  } catch (err) {
    console.error(err);
    fail(String(err.message || err));
  }
}

main();
