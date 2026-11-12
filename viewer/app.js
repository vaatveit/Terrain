// Two generated worlds on two globes, side by side, sharing one camera when linked.
//
// The data comes from scripts/view_worlds.py, which also serves this page. Each globe is the
// real Goldberg grid: every cell a 7-vertex fan (centre + 6 Voronoi corners), coloured in the
// shader from per-cell data textures. `gl_VertexID / 7` is the cell index, so no cell-index
// attribute is shipped. Both globes on one grid share the same position and index buffers on
// the GPU; a world contributes only its corner elevations and its field textures.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// Order is the shader's `uMode` numbering.
const MODES = [
  { key: 'elevation', label: 'elevation' },
  { key: 'koppen', label: 'köppen' },
  { key: 'biome', label: 'biome' },
  { key: 'temperature', label: 'temperature' },
  { key: 'precipitation', label: 'precipitation' },
  { key: 'discharge', label: 'discharge' },
  { key: 'uplift', label: 'uplift' },
  { key: 'plates', label: 'plates' },
  { key: 'landmasses', label: 'landmasses' },
];
const VERTS_PER_CELL = 7;
const INDICES_PER_CELL = 18;
const POS_SCALE = 32767;
// Cells per draw call. Every chunk reuses one index buffer of this size, which is what keeps
// an n=785 globe (6.2M cells) from needing a 444 MB index of its own; and cell ids run in
// spatially coherent blocks, so a chunk facing away from the camera can be skipped.
const CHUNK_CELLS = 1 << 19;
// Distinct colours for the ten largest landmasses; smaller ones share a grey.
const LANDMASS_COLOURS = [
  [0.902, 0.624, 0.000], [0.337, 0.706, 0.914], [0.000, 0.620, 0.451], [0.941, 0.894, 0.259],
  [0.000, 0.447, 0.698], [0.835, 0.369, 0.000], [0.800, 0.475, 0.655], [0.600, 0.800, 0.400],
  [0.700, 0.350, 0.350], [0.550, 0.500, 0.850],
];

// ---------------------------------------------------------------------------------------
// Shaders
// ---------------------------------------------------------------------------------------

const VERTEX = /* glsl */ `
precision highp sampler2D;
in float aElev;
in float aElevFlat;
uniform sampler2D uSA;      // elevation, temperature, precipitation, discharge
uniform sampler2D uSB;      // uplift, river width, lake level, landmass rank
uniform sampler2D uCodes;   // biome, flags (land|lake<<1|river<<2), koppen+1, plate
uniform int uTexW;
uniform int uCellOffset;    // first cell of this chunk
uniform float uExag;
uniform float uRadiusM;
uniform float uFlat;
uniform float uSeaFloor;
// Difference mode: the OTHER world's tables (world A's, when this is globe B), read at the
// same cell. Only ever bound between worlds on the same grid, where a cell id is a place.
uniform float uDiff;
uniform sampler2D uOtherSA;
uniform sampler2D uOtherSB;
uniform sampler2D uOtherCodes;

flat out int vCell;
flat out vec4 vA;
flat out vec4 vB;
flat out vec4 vC;
flat out vec4 vOA;
flat out vec4 vOB;
flat out vec4 vOC;
out vec3 vWorld;

void main() {
  int cell = uCellOffset + gl_VertexID / 7;
  vCell = cell;
  ivec2 uv = ivec2(cell % uTexW, cell / uTexW);
  vA = texelFetch(uSA, uv, 0);
  vB = texelFetch(uSB, uv, 0);
  vC = floor(texelFetch(uCodes, uv, 0) * 255.0 + 0.5);
  if (uDiff > 0.5) {
    vOA = texelFetch(uOtherSA, uv, 0);
    vOB = texelFetch(uOtherSB, uv, 0);
    vOC = floor(texelFetch(uOtherCodes, uv, 0) * 255.0 + 0.5);
  } else {
    vOA = vA; vOB = vB; vOC = vC;
  }
  int flags = int(vC.y);
  bool land = (flags & 1) != 0;
  bool lake = (flags & 2) != 0;

  float h = mix(aElev, aElevFlat, uFlat);
  if (lake) h = vB.z;                                  // lakes are flat at their level
  else if (!land) h = mix(max(h, 0.0), h, uSeaFloor);  // flood the sea to its surface
  vec3 p = normalize(position) * (1.0 + h * uExag / uRadiusM);
  vWorld = p;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
}
`;

const FRAGMENT = /* glsl */ `
precision highp sampler2D;
flat in int vCell;
flat in vec4 vA;
flat in vec4 vB;
flat in vec4 vC;
flat in vec4 vOA;
flat in vec4 vOB;
flat in vec4 vOC;
in vec3 vWorld;

uniform int uMode;
uniform float uDiff;
uniform float uDiffRange;    // the |A - B| that saturates the diverging ramp
uniform float uRivers;
uniform float uSeaFloor;
uniform vec2 uElevRange;     // deepest sea, highest land, over both worlds
uniform vec2 uRange;         // the current mode's range, over both worlds
uniform vec2 uRiverRange;
uniform vec3 uLightDir;
uniform sampler2D uBiomePal;
uniform sampler2D uKoppenPal;
uniform sampler2D uLandPal;
uniform float uLandPalSize;
uniform vec3 uRingDir;
uniform float uRingRadius;
uniform float uRingOn;

layout(location = 0) out vec4 fragColour;

const vec3 DEEP = vec3(0.027, 0.078, 0.180);
const vec3 SHALLOW = vec3(0.129, 0.353, 0.510);
const vec3 WATER = vec3(0.169, 0.404, 0.600);
// Water blues, not the near-white of a headwater: on these small planets river cells are
// a large share of the land, and pale ones read as snow on the elevation ramp.
const vec3 RIVER_THIN = vec3(0.420, 0.640, 0.840);
const vec3 RIVER_WIDE = vec3(0.140, 0.330, 0.560);

vec3 ramp(vec3 a, vec3 b, vec3 c, float t) {
  t = clamp(t, 0.0, 1.0);
  return t < 0.5 ? mix(a, b, t * 2.0) : mix(b, c, t * 2.0 - 1.0);
}
vec3 ramp5(vec3 a, vec3 b, vec3 c, vec3 d, vec3 e, float t) {
  float u = clamp(t, 0.0, 1.0) * 4.0;
  if (u < 1.0) return mix(a, b, u);
  if (u < 2.0) return mix(b, c, u - 1.0);
  if (u < 3.0) return mix(c, d, u - 2.0);
  return mix(d, e, u - 3.0);
}
float logNorm(float v, vec2 r) {
  float lo = log(max(r.x, 1e-12)), hi = log(max(r.y, 1e-12));
  return hi > lo ? clamp((log(max(v, 1e-12)) - lo) / (hi - lo), 0.0, 1.0) : 0.0;
}
vec3 hsv(float h, float s, float v) {
  vec3 k = clamp(abs(mod(h * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
  return v * mix(vec3(1.0), k, s);
}

vec3 sea(float e) {
  return mix(SHALLOW, DEEP, clamp(e / min(uElevRange.x, -1.0), 0.0, 1.0));
}

vec3 hypsometric(float e) {
  if (e < 0.0) return vec3(0.25, 0.36, 0.22);          // land below sea level
  return ramp5(vec3(0.247, 0.420, 0.243), vec3(0.580, 0.635, 0.373),
               vec3(0.741, 0.639, 0.420), vec3(0.545, 0.408, 0.322),
               vec3(0.960, 0.960, 0.970), e / max(uElevRange.y, 1.0));
}

vec3 cellColour() {
  int flags = int(vC.y);
  bool land = (flags & 1) != 0;
  bool lake = (flags & 2) != 0;
  bool river = (flags & 4) != 0;

  if (lake) return WATER;
  if (uRivers > 0.5 && river && vB.y > 0.0) {
    float t = uRiverRange.y > uRiverRange.x
      ? clamp(log(vB.y / uRiverRange.x) / log(uRiverRange.y / uRiverRange.x), 0.0, 1.0) : 0.0;
    return mix(RIVER_THIN, RIVER_WIDE, t);
  }

  if (uMode == 0) return land ? hypsometric(vA.x) : sea(vA.x);
  if (uMode == 7) {                                    // plates: over sea as well as land
    float h = fract(vC.w * 0.61803398875);
    vec3 c = hsv(h, 0.55, 0.9);
    return land ? c : mix(c, DEEP, 0.55);
  }
  if (!land) return sea(vA.x);
  if (uMode == 1) {
    int k = int(vC.z) - 1;
    return k < 0 ? DEEP : texelFetch(uKoppenPal, ivec2(k, 0), 0).rgb;
  }
  if (uMode == 2) return texelFetch(uBiomePal, ivec2(int(vC.x), 0), 0).rgb;
  float t = (uMode == 4 || uMode == 5) ? 0.0
          : clamp((vB.x - uRange.x) / max(uRange.y - uRange.x, 1e-6), 0.0, 1.0);
  if (uMode == 3) {
    t = clamp((vA.y - uRange.x) / max(uRange.y - uRange.x, 1e-6), 0.0, 1.0);
    return ramp(vec3(0.173, 0.353, 0.659), vec3(0.910, 0.910, 0.816),
                vec3(0.659, 0.196, 0.149), t);
  }
  if (uMode == 4) return ramp(vec3(0.690, 0.604, 0.369), vec3(0.184, 0.478, 0.227),
                              vec3(0.122, 0.373, 0.659), logNorm(vA.z, uRange));
  if (uMode == 5) return mix(vec3(0.106, 0.145, 0.196), vec3(0.345, 0.816, 0.910),
                             logNorm(vA.w, uRange));
  if (uMode == 6) {                                    // uplift: sinking blue, rising red
    float u = vB.x;
    if (u < 0.0) return mix(vec3(0.85), vec3(0.20, 0.35, 0.75),
                            clamp(u / min(uRange.x, -1e-3), 0.0, 1.0));
    return ramp(vec3(0.85), vec3(0.93, 0.62, 0.25), vec3(0.62, 0.08, 0.10),
                u / max(uRange.y, 1e-3));
  }
  // landmasses: the ten largest in fixed colours, smaller ones grey
  float r = vB.w;
  if (r >= 1.0 && r <= uLandPalSize) return texelFetch(uLandPal, ivec2(int(r) - 1, 0), 0).rgb;
  return vec3(0.50, 0.50, 0.48);
}

// The panel accents: orange for A, blue for B.
const vec3 ONLY_A = vec3(0.949, 0.639, 0.227);
const vec3 ONLY_B = vec3(0.298, 0.765, 0.910);
const vec3 SAME = vec3(0.52, 0.53, 0.55);
const vec3 CHANGED = vec3(0.92, 0.30, 0.62);

/** t in [-1, 1]: red where A is higher, blue where B is, near-white where they agree. */
vec3 diverge(float t) {
  t = clamp(t, -1.0, 1.0);
  return t >= 0.0 ? mix(vec3(0.93, 0.93, 0.91), vec3(0.70, 0.13, 0.13), t)
                  : mix(vec3(0.93, 0.93, 0.91), vec3(0.15, 0.33, 0.70), -t);
}

float logRatio(float a, float b, float floorV) {
  return log2(max(a, floorV) / max(b, floorV));
}

/** A - B at this cell, with vO* holding A and v* holding B. */
vec3 diffColour() {
  bool landA = (int(vOC.y) & 1) != 0;
  bool landB = (int(vC.y) & 1) != 0;
  float R = max(uDiffRange, 1e-9);
  if (uMode == 0) return diverge((vOA.x - vA.x) / R);         // elevation: land and sea floor
  if (uMode == 8) {                                            // landmasses: coastline change
    if (landA && landB) return SAME;
    if (landA) return ONLY_A;
    if (landB) return ONLY_B;
    return DEEP;
  }
  if (uMode == 7) return vOC.w == vC.w ? SAME : CHANGED;      // plates, sea included
  if (!landA && !landB) return DEEP;
  // Land in one world only: no like-for-like value, so say which world has it, dimmed.
  if (landA != landB) return mix(landA ? ONLY_A : ONLY_B, DEEP, 0.45);
  if (uMode == 1) return vOC.z == vC.z ? SAME : CHANGED;
  if (uMode == 2) return vOC.x == vC.x ? SAME : CHANGED;
  if (uMode == 3) return diverge((vOA.y - vA.y) / R);
  if (uMode == 4) return diverge(logRatio(vOA.z, vA.z, 1.0) / R);
  if (uMode == 5) return diverge(logRatio(vOA.w, vA.w, 1e-9) / R);
  return diverge((vOB.x - vB.x) / R);                          // uplift
}

void main() {
  // A true per-triangle normal from screen derivatives: the facets read as facets with no
  // normals shipped. The outward direction is radial, which settles the sign.
  vec3 N = normalize(cross(dFdx(vWorld), dFdy(vWorld)));
  if (dot(N, vWorld) < 0.0) N = -N;
  float lambert = max(dot(N, normalize(uLightDir)), 0.0);
  vec3 base = uDiff > 0.5 ? diffColour() : cellColour();
  vec3 colour = base * (0.34 + 0.72 * lambert);

  if (uRingOn > 0.5) {
    float a = acos(clamp(dot(normalize(vWorld), uRingDir), -1.0, 1.0));
    float w = uRingRadius * 0.22;
    if (abs(a - uRingRadius) < w) colour = vec3(1.0, 0.25, 0.85);
    else if (abs(a - uRingRadius) < w * 2.0) colour *= 0.25;
  }
  fragColour = vec4(colour, 1.0);
}
`;

const PICK_FRAGMENT = /* glsl */ `
flat in int vCell;
layout(location = 0) out vec4 fragColour;
void main() {
  fragColour = vec4(float(vCell & 255), float((vCell >> 8) & 255),
                    float((vCell >> 16) & 255), 0.0) / 255.0;
}
`;

// ---------------------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------------------

async function getJSON(url) {
  const r = await fetch(url);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.error || `${r.status} ${url}`);
  return body;
}

async function getBuffer(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  return r.arrayBuffer();
}

const geometries = new Map();   // key -> Promise<{ key, cells, dirs, chunks, index }>
let sharedIndex = null;

/** The fan pattern is the same for every cell, so one index buffer serves every chunk. */
function fanIndex() {
  if (!sharedIndex) {
    const idx = new Uint32Array(CHUNK_CELLS * INDICES_PER_CELL);
    for (let c = 0, w = 0; c < CHUNK_CELLS; c++) {
      const base = c * VERTS_PER_CELL;
      for (let k = 0; k < 6; k++) {
        idx[w++] = base;
        idx[w++] = base + 1 + k;
        idx[w++] = base + 1 + ((k + 1) % 6);
      }
    }
    sharedIndex = new THREE.BufferAttribute(idx, 1);
  }
  return sharedIndex;
}

function loadGeometry(key, cells) {
  if (!geometries.has(key)) {
    geometries.set(key, getBuffer(`/api/geometry/${encodeURIComponent(key)}`).then((buf) => {
      const dirs = new Int16Array(buf);
      if (dirs.length !== cells * VERTS_PER_CELL * 3) throw new Error(`geometry ${key} is the wrong size`);
      const chunks = [];
      for (let start = 0; start < cells; start += CHUNK_CELLS) {
        const count = Math.min(CHUNK_CELLS, cells - start);
        // The spherical cap the chunk's cell centres span, for skipping it when it faces away.
        const axis = new THREE.Vector3();
        for (let c = start; c < start + count; c++) {
          axis.x += dirs[c * 21]; axis.y += dirs[c * 21 + 1]; axis.z += dirs[c * 21 + 2];
        }
        axis.normalize();
        let minDot = 1;
        for (let c = start; c < start + count; c++) {
          const d = (axis.x * dirs[c * 21] + axis.y * dirs[c * 21 + 1] + axis.z * dirs[c * 21 + 2]) / POS_SCALE;
          if (d < minDot) minDot = d;
        }
        const sub = dirs.subarray(start * 21, (start + count) * 21);
        chunks.push({
          start, count, axis,
          cap: Math.acos(Math.max(-1, Math.min(1, minDot))),
          position: new THREE.Int16BufferAttribute(sub, 3, true),
        });
      }
      return { key, cells, dirs, chunks, index: fanIndex() };
    }));
  }
  return geometries.get(key);
}

function dataTexture(data, texW, type) {
  const tex = new THREE.DataTexture(data, texW, texW, THREE.RGBAFormat, type);
  tex.minFilter = tex.magFilter = THREE.NearestFilter;
  tex.generateMipmaps = false;
  tex.colorSpace = THREE.NoColorSpace;
  tex.needsUpdate = true;
  return tex;
}

function paletteTexture(colours) {
  const data = new Uint8Array(Math.max(1, colours.length) * 4);
  colours.forEach((c, i) => data.set([...c.map((v) => Math.round(v * 255)), 255], i * 4));
  const tex = new THREE.DataTexture(data, Math.max(1, colours.length), 1);
  tex.colorSpace = THREE.NoColorSpace;
  tex.needsUpdate = true;
  return tex;
}

function classPalette(rows) {
  const out = Array.from({ length: Math.max(...rows.map((r) => r.code)) + 1 }, () => [0, 0, 0]);
  for (const r of rows) out[r.code] = r.color;
  return out;
}

async function loadWorld(name, onStatus) {
  onStatus(`preparing ${name} … (the first view of a world exports it: seconds at n=256, ~10 s at n=785)`);
  const meta = await getJSON(`/api/world/${encodeURIComponent(name)}/meta.json`);
  onStatus(`loading ${name} (${(meta.cells / 1e6).toFixed(2)}M cells) …`);
  const [geom, buf] = await Promise.all([
    loadGeometry(meta.geometry, meta.cells),
    getBuffer(`/api/world/${encodeURIComponent(name)}/world.bin`),
  ]);
  const s = meta.sections;
  const cells = meta.cells;
  const texW = meta.texW;
  const view = (Ctor, sec) => new Ctor(buf, s[sec].offset, s[sec].count);
  // Copied out, so the corner elevations -- the other 40% of the buffer -- can be freed once
  // they reach the GPU. These three stay: the inspector reads them.
  const sA = view(Float32Array, 'scalarsA').slice();
  const sB = view(Float32Array, 'scalarsB').slice();
  const codes = view(Uint8Array, 'codes').slice();
  const elevSmooth = view(Int16Array, 'elevSmooth');
  const elevFlat = view(Int16Array, 'elevFlat');

  const own = {
    uSA: { value: dataTexture(sA, texW, THREE.FloatType) },
    uSB: { value: dataTexture(sB, texW, THREE.FloatType) },
    uCodes: { value: dataTexture(codes, texW, THREE.UnsignedByteType) },
    uTexW: { value: texW },
    uRadiusM: { value: meta.stats.radiusKm * 1000 },
    uDiff: { value: 0 },
  };
  // Bound to this world's own tables until difference mode points them at the other world's:
  // an unbound sampler is a link error even on a branch nothing takes.
  own.uOtherSA = { value: own.uSA.value };
  own.uOtherSB = { value: own.uSB.value };
  own.uOtherCodes = { value: own.uCodes.value };
  const free = function () { this.array = new Int16Array(0); };
  const group = new THREE.Group();
  const parts = geom.chunks.map((ch) => {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', ch.position);
    const e = new THREE.Int16BufferAttribute(elevSmooth.subarray(ch.start * 7, (ch.start + ch.count) * 7), 1, false);
    const ef = new THREE.Int16BufferAttribute(elevFlat.subarray(ch.start * 7, (ch.start + ch.count) * 7), 1, false);
    e.onUpload(free);
    ef.onUpload(free);
    geometry.setAttribute('aElev', e);
    geometry.setAttribute('aElevFlat', ef);
    geometry.setIndex(geom.index);
    geometry.setDrawRange(0, ch.count * INDICES_PER_CELL);
    geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1.5);
    const uniforms = () => ({ ...shared, ...own, uCellOffset: { value: ch.start } });
    const material = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3, uniforms: uniforms(), vertexShader: VERTEX, fragmentShader: FRAGMENT,
    });
    const pickMaterial = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3, uniforms: uniforms(), vertexShader: VERTEX, fragmentShader: PICK_FRAGMENT,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.frustumCulled = false;
    group.add(mesh);
    return { mesh, material, pickMaterial, axis: ch.axis, cap: ch.cap };
  });

  const present = (channel, offset) => {
    const seen = new Set();
    for (let i = 0; i < cells; i++) if (codes[i * 4 + 1] & 1) seen.add(codes[i * 4 + channel] - offset);
    return seen;
  };
  return {
    name, meta, geom, own, mesh: group, parts, sA, sB, codes,
    biomesPresent: present(0, 0), koppenPresent: present(2, 1),
    usePick(on) {
      for (const p of parts) p.mesh.material = on ? p.pickMaterial : p.material;
    },
    /** Hide the chunks over the horizon from `camera` -- about half the globe. */
    cull(camera) {
      const d = camera.position.length();
      const dir = camera.position.clone().normalize();
      const horizon = Math.acos(Math.min(1, 1 / Math.max(d, 1.0001)));
      for (const p of parts) p.mesh.visible = p.axis.angleTo(dir) < horizon + p.cap + 0.1;
    },
    dispose() {
      for (const p of parts) {
        p.mesh.geometry.dispose();
        p.material.dispose();
        p.pickMaterial.dispose();
      }
      for (const u of [own.uSA, own.uSB, own.uCodes]) u.value.dispose();
    },
  };
}


// ---------------------------------------------------------------------------------------
// Renderer, shared uniforms, panels
// ---------------------------------------------------------------------------------------

const canvas = document.getElementById('gl');
const stage = document.getElementById('stage');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
// Palettes arrive display-ready from the terrain modules; pass them through unconverted so the
// colours match render_world.py's PNGs.
renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
renderer.setScissorTest(true);

// The same IUniform objects sit in every material, so writing one moves both globes.
const shared = {
  uMode: { value: 0 },
  uExag: { value: 1 },
  uFlat: { value: 0 },
  uSeaFloor: { value: 0 },
  uRivers: { value: 1 },
  uElevRange: { value: new THREE.Vector2(-4000, 1000) },
  uRange: { value: new THREE.Vector2(0, 1) },
  uRiverRange: { value: new THREE.Vector2(1, 10) },
  uLightDir: { value: new THREE.Vector3(1, 0.4, 0.6).normalize() },
  uBiomePal: { value: paletteTexture([[0, 0, 0]]) },
  uKoppenPal: { value: paletteTexture([[0, 0, 0]]) },
  uLandPal: { value: paletteTexture(LANDMASS_COLOURS) },
  uLandPalSize: { value: LANDMASS_COLOURS.length },
  uRingDir: { value: new THREE.Vector3(1, 0, 0) },
  uRingRadius: { value: 0.02 },
  uRingOn: { value: 0 },
  uDiffRange: { value: 1 },
};

function makeCamera() {
  const cam = new THREE.PerspectiveCamera(40, 1, 0.002, 50);
  // Cell centres put the pole on +z and lon 0 on +x; up must be set before OrbitControls reads it.
  cam.up.set(0, 0, 1);
  cam.position.set(2.8, 0, 0.9);
  return cam;
}

function makeControls(camera, el) {
  const c = new OrbitControls(camera, el);
  c.enableDamping = true;
  c.dampingFactor = 0.1;
  c.enablePan = false;
  c.minDistance = 1.01;
  c.maxDistance = 8;
  c.rotateSpeed = 0.5;
  c.zoomSpeed = 0.9;
  return c;
}

const panels = ['a', 'b'].map((slot) => {
  const el = document.getElementById(`panel-${slot}`);
  const camera = makeCamera();
  return {
    slot, el, camera, ownCamera: camera,
    controls: makeControls(camera, el),
    scene: new THREE.Scene(),
    world: null,
    loading: null,
    status: el.querySelector('.status'),
    nameEl: el.querySelector('.name'),
    statsEl: el.querySelector('.stats'),
  };
});
const [pA, pB] = panels;

let linked = true;
function setLinked(on) {
  linked = on;
  pB.controls.dispose();
  if (on) {
    pB.camera = pA.camera;
  } else {
    pB.ownCamera.position.copy(pA.camera.position);
    pB.ownCamera.quaternion.copy(pA.camera.quaternion);
    pB.camera = pB.ownCamera;
  }
  pB.controls = makeControls(pB.camera, pB.el);
}
setLinked(true);

const visiblePanels = () => (pB.world || pB.loading ? panels : [pA]);

// ---------------------------------------------------------------------------------------
// UI state
// ---------------------------------------------------------------------------------------

const params = new URLSearchParams(location.search);
let modeIndex = Math.max(0, MODES.findIndex((m) => m.key === params.get('mode')));
let exagTouched = false;
let cameraFitted = false;
let diffWanted = params.get('diff') === '1';
let diffActive = false;   // wanted, and both worlds loaded on one grid

const selA = document.getElementById('world-a');
const selB = document.getElementById('world-b');
const modesEl = document.getElementById('modes');
const relief = document.getElementById('relief');
const reliefOut = document.getElementById('relief-out');
const legendEl = document.getElementById('legend');
const inspectorEl = document.getElementById('inspector');

for (const [i, m] of MODES.entries()) {
  const b = document.createElement('button');
  b.textContent = m.label;
  b.addEventListener('click', () => setMode(i));
  modesEl.append(b);
}

function setMode(i) {
  modeIndex = i;
  shared.uMode.value = i;
  [...modesEl.children].forEach((b, j) => b.classList.toggle('on', j === i));
  updateRanges();
  applyDiff();
  renderLegend();
  saveUrl();
}

function setExag(v) {
  shared.uExag.value = v;
  relief.value = v;
  reliefOut.textContent = `${Number(v).toFixed(1)}×`;
}
relief.addEventListener('input', () => { exagTouched = true; setExag(Number(relief.value)); });
document.getElementById('rivers').addEventListener('change', (e) => { shared.uRivers.value = e.target.checked ? 1 : 0; });
document.getElementById('flat').addEventListener('change', (e) => { shared.uFlat.value = e.target.checked ? 1 : 0; });
document.getElementById('seafloor').addEventListener('change', (e) => { shared.uSeaFloor.value = e.target.checked ? 1 : 0; });
document.getElementById('link').addEventListener('change', (e) => setLinked(e.target.checked));
const diffBox = document.getElementById('diff');
diffBox.checked = diffWanted;
diffBox.addEventListener('change', (e) => {
  diffWanted = e.target.checked;
  applyDiff();
  renderStats();
  renderLegend();
  renderInspector();
  saveUrl();
});
document.getElementById('swap').addEventListener('click', () => {
  const a = selA.value;
  const b = selB.value;
  if (!b) return;
  selA.value = b;
  selB.value = a;
  const wa = pA.world;
  pA.world = pB.world;
  pB.world = wa;
  for (const p of panels) {
    p.scene.clear();
    if (p.world) p.scene.add(p.world.mesh);
  }
  afterWorldsChanged();
});

function saveUrl() {
  const q = new URLSearchParams();
  if (selA.value) q.set('a', selA.value);
  if (selB.value) q.set('b', selB.value);
  q.set('mode', MODES[modeIndex].key);
  if (diffWanted) q.set('diff', '1');
  history.replaceState(null, '', `?${q}`);
}

// ---------------------------------------------------------------------------------------
// Loading worlds into panels
// ---------------------------------------------------------------------------------------

async function setWorld(panel, name) {
  const token = Symbol(name);
  panel.loading = token;
  panel.scene.clear();
  if (panel.world) panel.world.dispose();
  panel.world = null;
  applyDiff();               // never leave B sampling a world that has just been disposed
  panel.status.classList.remove('error');
  stage.classList.toggle('single', !(pB.world || pB.loading));
  if (!name) {
    panel.loading = null;
    panel.status.textContent = panel === pA ? 'pick a world' : '';
    afterWorldsChanged();
    return;
  }
  try {
    const world = await loadWorld(name, (text) => {
      if (panel.loading === token) panel.status.textContent = text;
    });
    if (panel.loading !== token) { world.dispose(); return; }
    panel.world = world;
    panel.scene.add(world.mesh);
    panel.status.textContent = '';
  } catch (err) {
    if (panel.loading !== token) return;
    panel.status.textContent = `failed: ${err.message}`;
    panel.status.classList.add('error');
    console.error(err);
  } finally {
    if (panel.loading === token) panel.loading = null;
  }
  afterWorldsChanged();
}

function loadedWorlds() {
  return panels.map((p) => p.world).filter(Boolean);
}

function afterWorldsChanged() {
  stage.classList.toggle('single', !(pB.world || pB.loading));
  const ws = loadedWorlds();
  if (ws.length) {
    shared.uBiomePal.value.dispose();
    shared.uKoppenPal.value.dispose();
    shared.uBiomePal.value = paletteTexture(classPalette(ws[0].meta.biomes));
    shared.uKoppenPal.value = paletteTexture(classPalette(ws[0].meta.koppen));
    if (!exagTouched) {
      // About 3% of the radius for the higher peak, so relief reads on any planet size.
      const r = Math.min(...ws.map((w) => w.meta.stats.radiusKm * 1000));
      const peak = Math.max(...ws.map((w) => Math.max(w.meta.ranges.landElevation[1], 1)));
      setExag(Math.min(10, Math.max(0.1, Math.round((0.03 * r) / peak * 10) / 10)));
    }
  }
  updateRanges();
  applyDiff();
  renderStats();
  renderLegend();
  saveUrl();
  resize();
  if (ws.length && !cameraFitted) { cameraFitted = true; fitCamera(); }
}

/** Back the camera off until the whole globe fits the narrower of its two field-of-view axes. */
function fitCamera() {
  const r = pA.el.getBoundingClientRect();
  const v = THREE.MathUtils.degToRad(pA.camera.fov);
  const h = 2 * Math.atan(Math.tan(v / 2) * (r.width / Math.max(r.height, 1)));
  const d = 1.08 / Math.sin(Math.min(v, h) / 2);
  for (const cam of [pA.ownCamera, pB.ownCamera]) cam.position.setLength(Math.min(d, 8));
}

/** Union over the loaded worlds, so a colour means the same value on both globes. */
function unionRange(field) {
  const ws = loadedWorlds();
  if (!ws.length) return [0, 1];
  return [Math.min(...ws.map((w) => w.meta.ranges[field][0])),
    Math.max(...ws.map((w) => w.meta.ranges[field][1]))];
}

function updateRanges() {
  const elev = unionRange('elevation');
  const land = unionRange('landElevation');
  shared.uElevRange.value.set(elev[0], land[1]);
  shared.uRiverRange.value.set(...unionRange('riverWidth'));
  const key = MODES[modeIndex].key;
  if (['temperature', 'precipitation', 'discharge', 'uplift'].includes(key)) {
    shared.uRange.value.set(...unionRange(key));
  }
}

// ---------------------------------------------------------------------------------------
// Difference mode: globe B recoloured as A - B, globe A left as the reference
// ---------------------------------------------------------------------------------------

const sameGrid = () => Boolean(pA.world && pB.world && pA.world.geom.key === pB.world.geom.key);

/** Point B's "other" samplers at A's tables (or back at its own), and fix the ramp's range. */
function applyDiff() {
  const can = sameGrid();
  diffBox.disabled = !can;
  diffBox.parentElement.title = can
    ? 'Recolour globe B as A − B, cell by cell'
    : 'Needs two worlds on the same grid (same n), where a cell id is the same place';
  diffActive = diffWanted && can;
  for (const p of panels) {
    const w = p.world;
    if (!w) continue;
    const src = diffActive && p === pB ? pA.world.own : w.own;
    w.own.uDiff.value = diffActive && p === pB ? 1 : 0;
    w.own.uOtherSA.value = src.uSA.value;
    w.own.uOtherSB.value = src.uSB.value;
    w.own.uOtherCodes.value = src.uCodes.value;
  }
  if (diffActive) shared.uDiffRange.value = diffRange(MODES[modeIndex].key);
}

/** The 98th percentile of |A - B| over a ~200k-cell sample: where the diverging ramp saturates.
 *  A percentile, not the maximum, so one freak cell cannot wash every other one out to white. */
function diffRange(key) {
  const A = pA.world;
  const B = pB.world;
  const n = A.meta.cells;
  const step = Math.max(1, Math.floor(n / 200000));
  const vals = [];
  for (let i = 0; i < n; i += step) {
    const both = (A.codes[i * 4 + 1] & 1) && (B.codes[i * 4 + 1] & 1);
    let d;
    if (key === 'elevation') d = A.sA[i * 4] - B.sA[i * 4];
    else if (!both) continue;
    else if (key === 'temperature') d = A.sA[i * 4 + 1] - B.sA[i * 4 + 1];
    else if (key === 'precipitation') d = Math.log2(Math.max(A.sA[i * 4 + 2], 1) / Math.max(B.sA[i * 4 + 2], 1));
    else if (key === 'discharge') d = Math.log2(Math.max(A.sA[i * 4 + 3], 1e-9) / Math.max(B.sA[i * 4 + 3], 1e-9));
    else if (key === 'uplift') d = A.sB[i * 4] - B.sB[i * 4];
    else return 1;
    vals.push(Math.abs(d));
  }
  if (!vals.length) return 1;
  vals.sort((a, b) => a - b);
  const q = vals[Math.floor(0.98 * (vals.length - 1))];
  return q > 0 ? q : Math.max(vals[vals.length - 1], 1e-6);
}

const DIVERGE = [[0.15, 0.33, 0.70], [0.93, 0.93, 0.91], [0.70, 0.13, 0.13]];
const ONLY_A = [0.949, 0.639, 0.227];
const ONLY_B = [0.298, 0.765, 0.910];
const DIM = (c) => c.map((v, i) => v * 0.55 + [0.027, 0.078, 0.180][i] * 0.45);

function renderDiffLegend() {
  const key = MODES[modeIndex].key;
  const R = shared.uDiffRange.value;
  const units = { elevation: ['m', 0], temperature: ['°C', 1], uplift: ['m/Myr', 1] };
  const mixed = [[DIM(ONLY_A), 'land in A only'], [DIM(ONLY_B), 'land in B only']];
  if (units[key]) {
    const [u, d] = units[key];
    gradient(`A − B ${key}: red where A is higher`, DIVERGE, `−${fmt(R, d)} ${u}`, `+${fmt(R, d)} ${u}`);
    if (key !== 'elevation') {
      legendEl.insertAdjacentHTML('beforeend',
        '<div class="title">land in one world only: orange for A, blue for B, dimmed</div>');
    }
  } else if (key === 'precipitation' || key === 'discharge') {
    const x = 2 ** R;
    gradient(`A ÷ B ${key}, log scale: red where A is ${key === 'precipitation' ? 'wetter' : 'higher'}`,
      DIVERGE, `B ${x.toFixed(1)}×`, `A ${x.toFixed(1)}×`);
  } else if (key === 'landmasses') {
    swatches('A − B coastlines', [[[0.52, 0.53, 0.55], 'land in both'], [ONLY_A, 'land in A only'], [ONLY_B, 'land in B only']]);
  } else {
    swatches(`A − B ${key}: where the class differs`,
      [[[0.52, 0.53, 0.55], 'same'], [[0.92, 0.30, 0.62], 'different']].concat(key === 'plates' ? [] : mixed));
  }
}

// ---------------------------------------------------------------------------------------
// Stats and legend
// ---------------------------------------------------------------------------------------

const pct = (v) => `${(v * 100).toFixed(v < 0.1 ? 1 : 0)}%`;
const fmt = (v, d = 0) => (v === null || v === undefined || Number.isNaN(v) ? '—' : Number(v).toFixed(d));

// [label, value, display, higher-is-"more" for the comparison highlight]
function statRows(s) {
  return [
    ['seed', s.seed, String(s.seed), null],
    ['preset', s.preset, s.preset, null],
    ['land', s.landFraction, pct(s.landFraction), true],
    ['large landmasses', s.largeLandmasses, String(s.largeLandmasses), true],
    ['largest', s.largestShare, pct(s.largestShare), null],
    ['peak', s.peakM, `${fmt(s.peakM)} m`, true],
    ['>300 m', s.above300, pct(s.above300), true],
    ['plates', s.plateCount, fmt(s.plateCount), null],
    ['belt', s.beltWidth, s.beltWidth ? `${fmt(s.beltWidth, 1)}×` : '—', null],
    ['hack h', s.hackH, fmt(s.hackH, 3), null],
  ];
}

function renderStats() {
  const both = pA.world && pB.world;
  for (const p of panels) {
    p.nameEl.textContent = p.world ? p.world.name : (p.loading ? '…' : '—');
    p.statsEl.replaceChildren();
    if (!p.world) continue;
    if (diffActive && p === pB) {
      p.nameEl.textContent = `A − B   (${pA.world.name} − ${pB.world.name}, on B's relief)`;
      continue;
    }
    const mine = statRows(p.world.meta.stats);
    const other = both ? statRows((p === pA ? pB : pA).world.meta.stats) : null;
    mine.forEach(([label, v, text, more], i) => {
      const d = document.createElement('span');
      d.className = 'stat';
      if (other && more !== null && typeof v === 'number' && v > other[i][1] * 1.0001) d.classList.add('better');
      d.innerHTML = `${label} <b>${text}</b>`;
      p.statsEl.append(d);
    });
  }
}

const rgb = (c) => `rgb(${c.map((v) => Math.round(v * 255)).join(',')})`;

function swatches(title, items) {
  legendEl.innerHTML = `<div class="title">${title}</div><div class="swatches"></div>`;
  const box = legendEl.querySelector('.swatches');
  for (const [colour, label] of items) {
    const s = document.createElement('span');
    s.className = 'sw';
    s.innerHTML = `<i style="background:${rgb(colour)}"></i>${label}`;
    box.append(s);
  }
}

function gradient(title, stops, lo, hi) {
  const css = stops.map((c, i) => `${rgb(c)} ${(i / (stops.length - 1)) * 100}%`).join(',');
  legendEl.innerHTML = `<div class="title">${title}</div>
    <div class="bar" style="background:linear-gradient(90deg,${css})"></div>
    <div class="ends"><span>${lo}</span><span>${hi}</span></div>`;
}

function renderLegend() {
  const ws = loadedWorlds();
  if (!ws.length) { legendEl.replaceChildren(); return; }
  if (diffActive) { renderDiffLegend(); return; }
  const key = MODES[modeIndex].key;
  const r = shared.uRange.value;
  const e = shared.uElevRange.value;
  if (key === 'koppen' || key === 'biome') {
    const rows = ws[0].meta[key === 'koppen' ? 'koppen' : 'biomes'];
    const seen = new Set(ws.flatMap((w) => [...(key === 'koppen' ? w.koppenPresent : w.biomesPresent)]));
    swatches(key === 'koppen' ? 'Köppen-Geiger (land classes on either globe)' : 'Whittaker biome',
      rows.filter((row) => seen.has(row.code)).map((row) => [row.color, row.name]));
  } else if (key === 'elevation') {
    gradient('land elevation, shared scale (sea: shallow → deep)',
      [[0.247, 0.420, 0.243], [0.580, 0.635, 0.373], [0.741, 0.639, 0.420], [0.545, 0.408, 0.322], [0.96, 0.96, 0.97]],
      '0 m', `${fmt(e.y)} m`);
  } else if (key === 'temperature') {
    gradient('mean annual temperature', [[0.173, 0.353, 0.659], [0.910, 0.910, 0.816], [0.659, 0.196, 0.149]],
      `${fmt(r.x, 1)} °C`, `${fmt(r.y, 1)} °C`);
  } else if (key === 'precipitation') {
    gradient('annual precipitation (log)', [[0.690, 0.604, 0.369], [0.184, 0.478, 0.227], [0.122, 0.373, 0.659]],
      `${fmt(r.x)} mm`, `${fmt(r.y)} mm`);
  } else if (key === 'discharge') {
    gradient('river discharge (log)', [[0.106, 0.145, 0.196], [0.345, 0.816, 0.910]],
      `${r.x.toExponential(1)} km³/yr`, `${r.y.toExponential(1)} km³/yr`);
  } else if (key === 'uplift') {
    gradient('tectonic uplift rate', [[0.20, 0.35, 0.75], [0.85, 0.85, 0.85], [0.93, 0.62, 0.25], [0.62, 0.08, 0.10]],
      `${fmt(r.x)} m/Myr`, `${fmt(r.y)} m/Myr`);
  } else if (key === 'plates') {
    legendEl.innerHTML = '<div class="title">tectonic plates — one colour each, dimmed over ocean</div>';
  } else {
    const shares = (w, i) => (w.meta.stats.landShares[i] !== undefined ? pct(w.meta.stats.landShares[i]) : '—');
    swatches('landmasses by size, share of land' + (ws.length > 1 ? ' (A / B)' : ''),
      LANDMASS_COLOURS.map((c, i) => [c, `#${i + 1} ${ws.map((w) => shares(w, i)).join(' / ')}`])
        .concat([[[0.5, 0.5, 0.48], 'smaller']]));
  }
}

// ---------------------------------------------------------------------------------------
// Hover: pick a cell, read it off both worlds, ring it on both globes
// ---------------------------------------------------------------------------------------

const pickTarget = new THREE.WebGLRenderTarget(1, 1, { type: THREE.UnsignedByteType, depthBuffer: true });
pickTarget.texture.colorSpace = THREE.NoColorSpace;
const pixel = new Uint8Array(4);
let pendingHover = null;
let lastPick = 0;
let hovered = null;   // { panel, cell, dir }

function pick(panel, x, y) {
  const w = panel.world;
  if (!w) return -1;
  const rect = panel.el.getBoundingClientRect();
  const dpr = renderer.getPixelRatio();
  const W = Math.max(1, Math.floor(rect.width * dpr));
  const H = Math.max(1, Math.floor(rect.height * dpr));
  const cam = panel.camera;
  cam.aspect = rect.width / rect.height;
  cam.setViewOffset(W, H, Math.floor(x * dpr), Math.floor(y * dpr), 1, 1);
  w.usePick(true);
  w.cull(cam);
  shared.uRingOn.value = 0;
  renderer.setRenderTarget(pickTarget);
  renderer.setClearColor(0x000000, 1);
  renderer.clear();
  renderer.render(panel.scene, cam);
  renderer.readRenderTargetPixels(pickTarget, 0, 0, 1, 1, pixel);
  renderer.setRenderTarget(null);
  cam.clearViewOffset();
  w.usePick(false);
  return pixel[3] === 255 ? -1 : pixel[0] | (pixel[1] << 8) | (pixel[2] << 16);
}

for (const p of panels) {
  p.el.addEventListener('pointermove', (e) => {
    const r = p.el.getBoundingClientRect();
    pendingHover = { panel: p, x: e.clientX - r.left, y: e.clientY - r.top };
  });
  p.el.addEventListener('pointerleave', () => {
    pendingHover = null;
    hovered = null;
    inspectorEl.hidden = true;
  });
}

function runPendingPick(now) {
  // A pick re-runs the vertex shader over the visible half of a globe: cheap at n=256, ~20M
  // vertices at n=785, so the big worlds are picked less often.
  const cells = pendingHover?.panel.world?.meta.cells ?? 0;
  if (!pendingHover || now - lastPick < (cells > 2e6 ? 120 : 50)) return;
  lastPick = now;
  const { panel, x, y } = pendingHover;
  pendingHover = null;
  const cell = pick(panel, x, y);
  if (cell < 0) { hovered = null; inspectorEl.hidden = true; return; }
  const d = panel.world.geom.dirs;
  const dir = new THREE.Vector3(d[cell * 21], d[cell * 21 + 1], d[cell * 21 + 2]).divideScalar(POS_SCALE).normalize();
  hovered = { panel, cell, dir };
  renderInspector();
}

function describe(w, cell) {
  const a = (i) => w.sA[cell * 4 + i];
  const b = (i) => w.sB[cell * 4 + i];
  const c = (i) => w.codes[cell * 4 + i];
  const flags = c(1);
  const land = flags & 1;
  const lake = flags & 2;
  const name = (rows, code) => rows.find((r) => r.code === code)?.name ?? '—';
  const rank = b(3);
  return {
    surface: lake ? 'lake' : land ? 'land' : 'sea',
    elevation: `${fmt(a(0))} m`,
    biome: land ? name(w.meta.biomes, c(0)) : '—',
    koppen: land && c(2) > 0 ? name(w.meta.koppen, c(2) - 1) : '—',
    temperature: `${fmt(a(1), 1)} °C`,
    precipitation: land ? `${fmt(a(2))} mm` : '—',
    discharge: land && a(3) > 0 ? `${a(3).toExponential(2)} km³/yr` : '—',
    'river width': b(1) > 0 ? `${fmt(b(1), 1)} m` : '—',
    uplift: `${fmt(b(0), 1)} m/Myr`,
    plate: String(c(3)),
    landmass: rank > 0 ? `#${rank} (${pct(w.meta.stats.landShares[rank - 1] ?? 0)} of land)` : '—',
  };
}

/** A - B at one cell, for the inspector's third column. Precipitation as a ratio, since it
 *  spans orders of magnitude; only where both worlds have land there. */
function deltas(A, B, cell) {
  const a = (w, i) => w.sA[cell * 4 + i];
  const b = (w, i) => w.sB[cell * 4 + i];
  const land = (w) => w.codes[cell * 4 + 1] & 1;
  const signed = (v, d, u) => `${v > 0 ? '+' : v < 0 ? '−' : '±'}${fmt(Math.abs(v), d)} ${u}`;
  const out = {
    elevation: signed(a(A, 0) - a(B, 0), 0, 'm'),
    temperature: signed(a(A, 1) - a(B, 1), 1, '°C'),
    uplift: signed(b(A, 0) - b(B, 0), 1, 'm/Myr'),
  };
  if (land(A) && land(B) && a(B, 2) > 0) out.precipitation = `×${(a(A, 2) / a(B, 2)).toFixed(2)}`;
  return out;
}

function renderInspector() {
  if (!hovered) { inspectorEl.hidden = true; return; }
  const { panel, cell, dir } = hovered;
  const cols = visiblePanels().filter((p) => p.world).map((p) => {
    // Same grid -> same cell id. A world on another grid has no cell of that id here.
    const same = p.world.geom.key === panel.world.geom.key;
    return { slot: p.slot, data: same ? describe(p.world, cell) : null };
  });
  const lat = (Math.asin(Math.max(-1, Math.min(1, dir.z))) * 180) / Math.PI;
  const lon = (Math.atan2(dir.y, dir.x) * 180) / Math.PI;
  const keys = Object.keys(cols.find((c) => c.data).data);
  // A - B beside the two columns, for the numeric fields, whenever the cell is the same place.
  const delta = cols.length === 2 && cols.every((c) => c.data) ? deltas(pA.world, pB.world, cell) : null;
  const head = cols.map((c) => `<td><span class="tag ${c.slot}">${c.slot.toUpperCase()}</span></td>`).join('')
    + (delta ? '<td>A − B</td>' : '');
  const row = (k) => cols.map((c) => `<td>${c.data ? c.data[k] : 'other grid'}</td>`).join('')
    + (delta ? `<td>${delta[k] ?? ''}</td>` : '');
  inspectorEl.innerHTML = `<div class="where">cell ${cell} · ${lat.toFixed(2)}°, ${lon.toFixed(2)}°</div>
    <table><thead><tr><th></th>${head}</tr></thead>
    <tbody>${keys.map((k) => `<tr><th>${k}</th>${row(k)}</tr>`).join('')}</tbody></table>`;
  inspectorEl.hidden = false;
}

// ---------------------------------------------------------------------------------------
// Frame loop
// ---------------------------------------------------------------------------------------

function resize() {
  const w = stage.clientWidth;
  const h = stage.clientHeight;
  renderer.setSize(w, h, false);
}
window.addEventListener('resize', resize);

const UP = new THREE.Vector3(0, 0, 1);
const light = new THREE.Vector3();

function frame(now) {
  for (const p of visiblePanels()) p.controls.update();
  runPendingPick(now);

  const stageRect = stage.getBoundingClientRect();
  const dpr = renderer.getPixelRatio();
  renderer.setClearColor(0x0b1017, 1);
  renderer.setScissor(0, 0, canvas.width, canvas.height);
  renderer.setViewport(0, 0, canvas.width / dpr, canvas.height / dpr);
  renderer.clear();

  for (const p of visiblePanels()) {
    if (!p.world) continue;
    const r = p.el.getBoundingClientRect();
    const left = r.left - stageRect.left;
    const bottom = stageRect.bottom - r.bottom;
    renderer.setViewport(left, bottom, r.width, r.height);
    renderer.setScissor(left, bottom, r.width, r.height);
    p.camera.aspect = r.width / r.height;
    p.camera.updateProjectionMatrix();

    // A raking light that follows the camera: a fixed sun leaves half the planet dark and a
    // pure headlight flattens the relief.
    light.copy(p.camera.position).normalize().applyAxisAngle(UP, 0.7);
    light.z += 0.35;
    shared.uLightDir.value.copy(light).normalize();

    if (hovered) {
      shared.uRingOn.value = 1;
      shared.uRingDir.value.copy(hovered.dir);
      shared.uRingRadius.value = 0.003 + 0.012 * (p.camera.position.length() - 1);
    } else {
      shared.uRingOn.value = 0;
    }
    p.world.cull(p.camera);
    renderer.render(p.scene, p.camera);
  }
  requestAnimationFrame(frame);
}

// ---------------------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------------------

async function start() {
  const { worlds, defaults } = await getJSON('/api/worlds');
  // One option group per folder the server lists worlds from.
  const groups = new Map();
  for (const w of worlds) {
    if (!groups.has(w.group)) groups.set(w.group, []);
    groups.get(w.group).push(w.name);
  }
  const fill = (sel, allowNone) => {
    sel.replaceChildren(new Option(allowNone ? '— none —' : '— pick a world —', ''));
    for (const [group, names] of groups) {
      const og = document.createElement('optgroup');
      og.label = group;
      for (const n of names) og.append(new Option(n, n));
      sel.append(og);
    }
  };
  const names = worlds.map((w) => w.name);
  fill(selA, false);
  fill(selB, true);
  const a = params.get('a') ?? defaults[0] ?? '';
  const b = params.get('b') ?? defaults[1] ?? '';
  selA.value = names.includes(a) ? a : '';
  selB.value = names.includes(b) ? b : '';
  selA.addEventListener('change', () => setWorld(pA, selA.value));
  selB.addEventListener('change', () => setWorld(pB, selB.value));

  setMode(modeIndex);
  resize();
  requestAnimationFrame(frame);
  await Promise.all([setWorld(pA, selA.value), setWorld(pB, selB.value)]);
}

// Console handle for debugging.
window.viewer = { THREE, renderer, panels, shared, pick };

start().catch((err) => {
  pA.status.textContent = `failed: ${err.message}`;
  pA.status.classList.add('error');
  console.error(err);
});
