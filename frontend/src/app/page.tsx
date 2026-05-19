"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type CatalogRow = {
  name: string;
  family: string;
  domain: string;
  coordinate_system: string;
  resolution: string;
  n_steps: string;
  n_traj: string;
  size_gb: string;
  fields: string;
  useful_for: string;
  not_useful_for: string;
  source_url: string;
};

type InterpretationRow = {
  field: string;
  value: string;
  why_it_matters: string;
};

type FrameStat = {
  frame: number;
  min: number;
  max: number;
  mean: number;
  std: number;
  gradient_energy: number;
};

type SourceRow = {
  label: string;
  value: string;
  note: string;
};

type AuthoringBrief = {
  title?: string;
  anchor_dataset?: string;
  route_reason?: string;
  visual_goal?: string;
  scene_beats?: string[];
  labels?: string[];
  palette?: string[];
  camera?: string[];
  narration?: string[];
  safety_notes?: string[];
  manim_prompt?: string;
};

type ExplainerStatus = {
  job_id: string;
  status: string;
  url?: string;
  route?: Record<string, unknown>;
  brief?: AuthoringBrief;
  manim_prompt?: string;
  authoring?: AuthoringBrief;
  label?: string;
  attempts?: number;
  provider?: string;
  reason?: string;
  explanation?: string;
  prompt?: string;
};

type SimResponse = {
  task: string;
  frames: number[][][];
  meta: Record<string, unknown>;
  route: Record<string, unknown>;
  interpretation: InterpretationRow[];
  llm_explanation?: Record<string, unknown>;
  warning: string | null;
};

type ModelInfo = {
  checkpoint_loaded: boolean;
  checkpoint_path: string;
  meta: Record<string, unknown>;
  training_report?: { best_val_loss?: number; history?: Array<Record<string, number>>; inventory?: Record<string, unknown> } | null;
  catalog_summary?: Record<string, unknown>;
  catalog: CatalogRow[];
  simulation_modes?: Array<Record<string, unknown>>;
  datasets_trained: Array<Record<string, unknown>>;
  interpretation: Record<string, string>;
  llm_explanation?: Record<string, unknown>;
};

const API = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";
const DEFAULT_STEPS = 96;
const PLAYBACK_MS = 58;

type Preset = {
  label: string;
  task: string;
  mode: string;
};

type PreloadState = {
  loaded: number;
  total: number;
  active: string;
};

const PRESETS: Preset[] = [
  { label: "#0 MHD", task: "generate an MHD plasma simulation: supersonic sub-Alfvenic wave turbulence", mode: "field" },
  { label: "#1 BLACK_HOLE", task: "create a black hole accretion disk in deep space, environment built by reactor.inc", mode: "intensity" },
  { label: "#2 SUPERNOVA", task: "predict a supernova blast wave through turbulent interstellar gas", mode: "shock_front" },
  { label: "#3 MAZE_WAVE", task: "simulate acoustic wave scattering through a maze", mode: "pressure" },
  { label: "#4 CONVECTION", task: "show Rayleigh-Benard thermal convection plume", mode: "temperature" },
  { label: "#5 GRAY_SCOTT", task: "generate Gray-Scott chemical reaction diffusion pattern formation", mode: "concentration" },
  { label: "#6 SHEAR", task: "simulate Kelvin-Helmholtz shear flow mixing layer", mode: "vorticity" },
  { label: "#7 PLANET", task: "predict planetary shallow-water storm vortex and jet stream", mode: "height" },
  { label: "#8 MHD_SHOCK", task: "generate magnetized shock front in turbulent plasma", mode: "shock_edges" }
];

const FALLBACK_MODES: Record<string, string[]> = {
  black_hole: ["intensity", "heat_radiation", "lensing", "doppler"],
  supernova: ["density", "shock_front", "temperature", "ejecta"],
  acoustic: ["pressure", "wave_energy", "maze_geometry", "interference"],
  convection: ["temperature", "heat_flux", "plume_velocity", "rolls"],
  reaction: ["concentration", "reaction_rate", "pattern_edges"],
  shear: ["scalar", "vorticity", "mixing", "velocity"],
  planetary: ["height", "vorticity", "jet_stream", "storm_track"],
  mhd_scalar: ["field", "gradient", "shock_edges", "magnetic_proxy"]
};

function colorMap(value: number, gradient: number): [number, number, number] {
  const v = Math.max(0, Math.min(1, value));
  const boost = Math.max(0, Math.min(0.55, gradient * 3.5));
  const stops: Array<[number, number, number, number]> = [
    [0.0, 4, 8, 30],
    [0.15, 20, 45, 130],
    [0.34, 10, 135, 190],
    [0.56, 70, 220, 120],
    [0.76, 255, 210, 70],
    [1.0, 245, 42, 34]
  ];
  for (let i = 1; i < stops.length; i++) {
    const [p1, r1, g1, b1] = stops[i];
    const [p0, r0, g0, b0] = stops[i - 1];
    if (v <= p1) {
      const t = (v - p0) / Math.max(0.0001, p1 - p0);
      return [
        Math.min(255, Math.round(r0 + (r1 - r0) * t + boost * 95)),
        Math.min(255, Math.round(g0 + (g1 - g0) * t + boost * 95)),
        Math.min(255, Math.round(b0 + (b1 - b0) * t + boost * 120))
      ];
    }
  }
  return [245, 42, 34];
}

function visualColor(value: number, gradient: number, style: string): [number, number, number] {
  const v = Math.max(0, Math.min(1, value));
  if (style === "black_hole") {
    if (v < 0.025) return [0, 0, 3];
    const hot = Math.pow(v, 0.62);
    return [
      Math.min(255, Math.round(255 * hot)),
      Math.min(255, Math.round(118 * hot + 120 * Math.max(0, v - 0.72))),
      Math.min(255, Math.round(18 * hot + 170 * Math.max(0, v - 0.86)))
    ];
  }
  if (style === "supernova") {
    return [
      Math.min(255, Math.round(35 + 245 * v)),
      Math.min(255, Math.round(8 + 170 * Math.pow(v, 1.7))),
      Math.min(255, Math.round(30 + 90 * (1 - v) + 90 * Math.max(0, v - 0.75)))
    ];
  }
  if (style === "acoustic") {
    const c = Math.round(30 + 225 * v);
    return [Math.round(c * 0.72), Math.round(c * 0.9), c];
  }
  if (style === "convection") {
    return [
      Math.min(255, Math.round(25 + 245 * v)),
      Math.min(255, Math.round(40 + 160 * (1 - Math.abs(v - 0.55)))),
      Math.min(255, Math.round(190 * (1 - v)))
    ];
  }
  if (style === "reaction") {
    return [
      Math.min(255, Math.round(30 + 70 * v)),
      Math.min(255, Math.round(15 + 240 * v)),
      Math.min(255, Math.round(45 + 180 * (1 - v)))
    ];
  }
  if (style === "shear") {
    return [
      Math.min(255, Math.round(30 + 210 * v)),
      Math.min(255, Math.round(30 + 210 * (1 - Math.abs(v - 0.5) * 1.6))),
      Math.min(255, Math.round(220 * (1 - v)))
    ];
  }
  if (style === "planetary") {
    return [
      Math.min(255, Math.round(12 + 70 * v + 120 * Math.max(0, v - 0.74))),
      Math.min(255, Math.round(45 + 165 * v)),
      Math.min(255, Math.round(90 + 145 * (1 - Math.abs(v - 0.45))))
    ];
  }
  return colorMap(v, gradient);
}

function energy(frame: number[][]): number {
  if (!frame.length) return 0;
  let total = 0;
  let count = 0;
  for (let y = 1; y < frame.length; y++) {
    for (let x = 1; x < frame[y].length; x++) {
      total += Math.abs(frame[y][x] - frame[y][x - 1]) + Math.abs(frame[y][x] - frame[y - 1][x]);
      count += 2;
    }
  }
  return total / Math.max(1, count);
}

function cell(value: unknown): string {
  if (value === null || value === undefined || value === "") return "unknown";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(6);
  return String(value);
}


function isExplainerTerminal(status?: string): boolean {
  return status === "done" || status === "done_fallback" || status === "error";
}

function mediaUrl(path?: string): string {
  if (!path) return "";
  if (/^https?:\/\//.test(path)) return path;
  return API + path;
}
function RawBlock({ value }: { value: unknown }) {
  return <pre className="raw">{JSON.stringify(value, null, 2)}</pre>;
}

function sampleFrame(frame: number[][], size = 10): number[][] {
  if (!frame.length || !frame[0]?.length) return [];
  const yStep = Math.max(1, Math.floor(frame.length / size));
  const xStep = Math.max(1, Math.floor(frame[0].length / size));
  const rows: number[][] = [];
  for (let y = 0; y < frame.length && rows.length < size; y += yStep) {
    const row: number[] = [];
    for (let x = 0; x < frame[y].length && row.length < size; x += xStep) {
      row.push(frame[y][x]);
    }
    rows.push(row);
  }
  return rows;
}

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const cacheRef = useRef<Map<string, SimResponse>>(new Map());
  const inflightRef = useRef<Map<string, Promise<SimResponse>>>(new Map());
  const requestSeqRef = useRef(0);
  const bootStartedRef = useRef(false);
  const [task, setTask] = useState(PRESETS[0].task);
  const [mode, setMode] = useState(PRESETS[0].mode);
  const [steps, setSteps] = useState(DEFAULT_STEPS);
  const [response, setResponse] = useState<SimResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const [running, setRunning] = useState(false);
  const [activeTab, setActiveTab] = useState("RUN");
  const [error, setError] = useState("");
  const [loadingSimulation, setLoadingSimulation] = useState(true);
  const [preload, setPreload] = useState<PreloadState>({ loaded: 0, total: PRESETS.length, active: "idle" });
  const [explainerPrompt, setExplainerPrompt] = useState("explain magnetic reconnection in a plasma");
  const [explainerJobId, setExplainerJobId] = useState("");
  const [explainerState, setExplainerState] = useState<ExplainerStatus | null>(null);
  const [explainerLoading, setExplainerLoading] = useState(false);
  const [explainerError, setExplainerError] = useState("");

  const frame = response?.frames?.[frameIndex] ?? [];
  const frameStats = (response?.meta?.frame_stats as FrameStat[] | undefined) ?? [];
  const currentStat = frameStats[frameIndex] ?? frameStats[0];
  const frameSample = useMemo(() => sampleFrame(frame), [frame]);
  const waveEnergy = useMemo(() => energy(frame), [frame]);
  const catalog = modelInfo?.catalog ?? [];
  const loadedDataset = String(response?.meta?.dataset ?? modelInfo?.meta?.dataset ?? "MHD_64");
  const routedDataset = String(response?.meta?.dataset_hint ?? response?.route?.recommended_dataset ?? "MHD_64");
  const visualStyle = String(response?.meta?.visual_style ?? "mhd_scalar");
  const selectedMode = String(response?.meta?.mode ?? mode);
  const supportedModes = useMemo(() => {
    const fromResponse = response?.meta?.supported_modes;
    if (Array.isArray(fromResponse) && fromResponse.length) return fromResponse.map(String);
    return FALLBACK_MODES[visualStyle] ?? FALLBACK_MODES.mhd_scalar;
  }, [response, visualStyle]);
  const selectedCheckpointLoaded = response ? response.meta?.checkpoint_loaded === true : modelInfo?.checkpoint_loaded === true;
  const trainedForRequest = response?.meta?.trained_for_request;
  const history = modelInfo?.training_report?.history ?? [];
  const inventory = modelInfo?.training_report?.inventory as Record<string, unknown> | undefined;
  const budget = inventory?.training_budget as Record<string, unknown> | undefined;
  const trainedRows = modelInfo?.datasets_trained ?? [];
  const simulationModes = modelInfo?.simulation_modes ?? [];
  const sourceRows = (response?.meta?.data_source_rows as SourceRow[] | undefined) ?? [];
  const visibleInterpretation = response?.interpretation?.slice(0, 6) ?? [];
  const explainerVideoSrc = mediaUrl(explainerState?.url);
  const authoringBrief = (explainerState?.brief ?? explainerState?.authoring) as AuthoringBrief | undefined;
  const manimPrompt = String(explainerState?.manim_prompt ?? authoringBrief?.manim_prompt ?? "");
  const promptLines = manimPrompt ? manimPrompt.split(/\r?\n/).slice(0, 12) : [];
  const trainingNotice = response?.warning
    ?? (!selectedCheckpointLoaded && response ? `NO LOCAL ${routedDataset} CHECKPOINT; DISPLAYING ${cell(response.meta?.data_source_kind)} INSTEAD.` : "");

  function localDatasetStatus(name: string): string {
    const entry = simulationModes.find((row) => String(row.dataset) === name);
    if (!entry) return name === loadedDataset && modelInfo?.checkpoint_loaded ? "TRAINED_CHECKPOINT" : "CATALOG_ONLY";
    const checkpoint = entry.checkpoint_exists === true ? "CHECKPOINT_AVAILABLE" : "CHECKPOINT_MISSING";
    return `${checkpoint}; FALLBACK=${cell(entry.fallback)}`;
  }

  function cacheKey(selectedTask: string, selectedMode: string, selectedSteps: number): string {
    return `${selectedSteps}\n${selectedMode}\n${selectedTask}`;
  }

  async function loadSimulationPayload(selectedTask: string, requestedMode: string, requestedSteps: number): Promise<SimResponse> {
    const key = cacheKey(selectedTask, requestedMode, requestedSteps);
    const cached = cacheRef.current.get(key);
    if (cached) return cached;
    const inflight = inflightRef.current.get(key);
    if (inflight) return inflight;

    const promise = fetch(API + "/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task: selectedTask, steps: requestedSteps, mode: requestedMode })
    }).then(async (res) => {
      if (!res.ok) throw new Error(await res.text());
      const payload = (await res.json()) as SimResponse;
      cacheRef.current.set(key, payload);
      return payload;
    }).finally(() => {
      inflightRef.current.delete(key);
    });

    inflightRef.current.set(key, promise);
    return promise;
  }

  function commitSimulation(payload: SimResponse, requestedMode: string) {
    setResponse(payload);
    setMode(String(payload.meta?.mode ?? requestedMode));
    setFrameIndex(0);
    setRunning(Boolean(payload.frames.length));
    setActiveTab("INTERPRET");
  }

  async function warmPresetCache(selectedSteps: number, initialPayload: SimResponse, initialMode: string) {
    cacheRef.current.set(cacheKey(initialPayload.task, initialMode, selectedSteps), initialPayload);
    let loaded = 1;
    setPreload({ loaded, total: PRESETS.length, active: "boot" });
    for (const preset of PRESETS) {
      const key = cacheKey(preset.task, preset.mode, selectedSteps);
      if (cacheRef.current.has(key)) continue;
      setPreload({ loaded, total: PRESETS.length, active: preset.label });
      try {
        await loadSimulationPayload(preset.task, preset.mode, selectedSteps);
        loaded += 1;
      } catch {
        // Keep the foreground demo usable even if one background preset fails.
      }
      setPreload({ loaded, total: PRESETS.length, active: preset.label });
    }
    setPreload({ loaded, total: PRESETS.length, active: "ready" });
  }

  useEffect(() => {
    if (bootStartedRef.current) return;
    bootStartedRef.current = true;
    let cancelled = false;
    async function boot() {
      setLoadingSimulation(true);
      setError("");
      try {
        const params = new URLSearchParams(window.location.search);
        const presetIndex = Number(params.get("preset") ?? "0");
        const preset = PRESETS[presetIndex] ?? PRESETS[0];
        const bootTask = params.get("task") ?? preset.task;
        const bootMode = params.get("mode") ?? preset.mode;
        const bootSteps = Math.max(1, Math.min(256, Number(params.get("steps") ?? DEFAULT_STEPS)));
        setTask(bootTask);
        setMode(bootMode);
        setSteps(bootSteps);
        const [modelRes, simPayload] = await Promise.all([
          fetch(API + "/model/info"),
          loadSimulationPayload(bootTask, bootMode, bootSteps)
        ]);
        if (!modelRes.ok) throw new Error(await modelRes.text());
        const modelPayload = (await modelRes.json()) as ModelInfo;
        if (cancelled) return;
        setModelInfo(modelPayload);
        commitSimulation(simPayload, bootMode);
        void warmPresetCache(bootSteps, simPayload, bootMode);
      } catch (err) {
        if (!cancelled) setError(String(err));
      } finally {
        if (!cancelled) setLoadingSimulation(false);
      }
    }
    void boot();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!explainerJobId || isExplainerTerminal(explainerState?.status)) return;
    let cancelled = false;

    async function pollExplainer() {
      try {
        const res = await fetch("/api/explainer/status/" + explainerJobId);
        if (!res.ok) throw new Error(await res.text());
        const payload = (await res.json()) as ExplainerStatus;
        if (cancelled) return;
        setExplainerState(payload);
        if (isExplainerTerminal(payload.status)) setExplainerLoading(false);
      } catch (err) {
        if (!cancelled) {
          setExplainerError(String(err));
          setExplainerLoading(false);
        }
      }
    }

    void pollExplainer();
    const timer = window.setInterval(pollExplainer, 2000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [explainerJobId, explainerState?.status]);
  useEffect(() => {
    if (!running || !response?.frames?.length) return;
    let raf = 0;
    let last = performance.now();
    let carry = 0;
    const frameCount = response.frames.length;
    const tick = (now: number) => {
      carry += now - last;
      last = now;
      if (carry >= PLAYBACK_MS) {
        const advance = Math.max(1, Math.floor(carry / PLAYBACK_MS));
        carry %= PLAYBACK_MS;
        setFrameIndex((current) => (current + advance) % frameCount);
      }
      raf = window.requestAnimationFrame(tick);
    };
    raf = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(raf);
  }, [running, response?.frames?.length]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !frame.length) return;
    const height = frame.length;
    const width = frame[0].length;
    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.imageSmoothingEnabled = true;
    const image = ctx.createImageData(width, height);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const here = frame[y][x];
        const gx = x > 0 ? Math.abs(here - frame[y][x - 1]) : 0;
        const gy = y > 0 ? Math.abs(here - frame[y - 1][x]) : 0;
        const index = (y * width + x) * 4;
        const [r, g, b] = visualColor(here, gx + gy, visualStyle);
        image.data[index] = r;
        image.data[index + 1] = g;
        image.data[index + 2] = b;
        image.data[index + 3] = 255;
      }
    }
    ctx.putImageData(image, 0, 0);
  }, [frame, visualStyle]);

  async function runSimulation(taskOverride?: string, modeOverride?: string) {
    const selectedTask = taskOverride ?? task;
    const requestedMode = modeOverride ?? mode;
    const requestedSteps = Math.max(1, Math.min(256, steps));
    const seq = requestSeqRef.current + 1;
    requestSeqRef.current = seq;
    setTask(selectedTask);
    setMode(requestedMode);
    setError("");
    setLoadingSimulation(true);
    setRunning(false);
    setFrameIndex(0);
    try {
      const payload = await loadSimulationPayload(selectedTask, requestedMode, requestedSteps);
      if (seq !== requestSeqRef.current) return;
      commitSimulation(payload, requestedMode);
    } catch (err) {
      if (seq === requestSeqRef.current) setError(String(err));
    } finally {
      if (seq === requestSeqRef.current) setLoadingSimulation(false);
    }
  }
  async function runExplainer(promptOverride?: string) {
    const selectedPrompt = (promptOverride ?? explainerPrompt).trim();
    if (!selectedPrompt) return;
    setExplainerPrompt(selectedPrompt);
    setExplainerError("");
    setExplainerLoading(true);
    setExplainerState({ job_id: "", status: "queued", prompt: selectedPrompt });
    setActiveTab("EXPLAINER");
    try {
      const res = await fetch("/api/explainer/animate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: selectedPrompt })
      });
      if (!res.ok) throw new Error(await res.text());
      const payload = (await res.json()) as ExplainerStatus;
      setExplainerJobId(payload.job_id);
      setExplainerState(payload);
      if (isExplainerTerminal(payload.status)) setExplainerLoading(false);
    } catch (err) {
      setExplainerError(String(err));
      setExplainerLoading(false);
    }
  }

  return (
    <main>
      <pre className="title">{`; ANIMATED_EXPLAINER :: WORKBENCH
; BACKEND=${API}
; LOADED_DATASET=${loadedDataset}   ROUTED_TASK=${routedDataset}   VISUAL_STYLE=${visualStyle}   MODE=${selectedMode}
; FLOW=BRIEF -> PROMPT -> RENDER -> FALLBACK`}</pre>

      <section className="block sim-block">
        <div className="label">SECTION .SIMULATION_TOP</div>
        <canvas ref={canvasRef} className="sim-canvas" aria-label="color simulation frame" />
        {!frame.length && <div className="empty">{loadingSimulation ? "LOADING REAL WELL CHECKPOINT FRAMES..." : "FRAME_BUFFER_EMPTY :: PRESS INT 0x80 RUN"}</div>}
        <div className="row statusline">
          <span>{response || modelInfo ? (selectedCheckpointLoaded ? "JNZ ROUTE_CHECKPOINT_LOADED" : "JZ ROUTE_CHECKPOINT_MISSING") : "WAIT CHECKPOINT_STATUS"}</span>
          <span>FRAME {response?.frames?.length ? frameIndex + 1 : 0}/{response?.frames?.length ?? 0}</span>
          <span>SIZE {cell(response?.meta?.frame_size ?? modelInfo?.meta?.frame_size)}</span>
          <span>WAVE_GRAD {waveEnergy.toFixed(6)}</span>
          <span>LOADED {loadedDataset}</span>
          <span>ROUTE {routedDataset}</span>
          <span>STYLE {visualStyle}</span>
          <span>MODE {selectedMode}</span>
          <span>CACHE {preload.loaded}/{preload.total} {preload.active}</span>
          <span>{trainedForRequest === false ? "MISMATCH_ROUTE" : "ROUTE_OK_OR_IDLE"}</span>
        </div>
      </section>

      <section className="visible-grid">
        <div className="block">
          <div className="label">SECTION .VISIBLE_SIM_DATA</div>
          <table>
            <tbody>
              <tr><th>frames_loaded</th><td>{cell(response?.frames?.length ?? 0)}</td></tr>
              <tr><th>frame_shape</th><td>{frame.length ? `${frame.length} x ${frame[0].length}` : "loading"}</td></tr>
              <tr><th>loaded_checkpoint</th><td>{loadedDataset}</td></tr>
              <tr><th>routed_dataset</th><td>{routedDataset}</td></tr>
              <tr><th>simulation_kind</th><td>{cell(response?.meta?.simulation_kind)}</td></tr>
              <tr><th>visual_style</th><td>{visualStyle}</td></tr>
              <tr><th>mode</th><td>{selectedMode}</td></tr>
              <tr><th>supported_modes</th><td>{supportedModes.join(", ")}</td></tr>
              <tr><th>data_source_kind</th><td>{cell(response?.meta?.data_source_kind)}</td></tr>
              <tr><th>trained_for_request</th><td>{cell(trainedForRequest ?? "pending")}</td></tr>
              <tr><th>prediction_method</th><td>{cell(response?.meta?.prediction_method)}</td></tr>
              <tr><th>current_mean</th><td>{cell(currentStat?.mean)}</td></tr>
              <tr><th>current_std</th><td>{cell(currentStat?.std)}</td></tr>
              <tr><th>gradient_energy</th><td>{cell(currentStat?.gradient_energy ?? waveEnergy)}</td></tr>
            </tbody>
          </table>
        </div>

        <div className="block">
          <div className="label">SECTION .FRAME_MATRIX_SAMPLE</div>
          <table className="matrix">
            <tbody>
              {frameSample.map((row, y) => (
                <tr key={y}>
                  {row.map((value, x) => <td key={x}>{value.toFixed(2)}</td>)}
                </tr>
              ))}
              {!frameSample.length && <tr><td>waiting for frame buffer</td></tr>}
            </tbody>
          </table>
        </div>
      </section>

      <section className="visible-grid">
        <div className="block">
          <div className="label">SECTION .VISIBLE_SOURCE_DATA</div>
          <table>
            <thead><tr><th>label</th><th>value</th><th>note</th></tr></thead>
            <tbody>
              {sourceRows.map((row, index) => (
                <tr key={index}>
                  <td>{cell(row.label)}</td>
                  <td>{cell(row.value)}</td>
                  <td>{cell(row.note)}</td>
                </tr>
              ))}
              {!sourceRows.length && trainedRows.map((row, index) => (
                <tr key={index}>
                  <td>{cell(row.name)}</td>
                  <td>train={cell(row.train_items_available)} valid={cell(row.validation_items_available)}</td>
                  <td>{cell(row.field_extraction)} frame={cell(row.emulator_frame_size)}</td>
                </tr>
              ))}
              {!sourceRows.length && !trainedRows.length && <tr><td colSpan={3}>loading selected simulation source rows</td></tr>}
            </tbody>
          </table>
        </div>

        <div className="block">
          <div className="label">SECTION .VISIBLE_INTERPRETATION</div>
          <table>
            <thead><tr><th>field</th><th>value</th><th>why</th></tr></thead>
            <tbody>
              {visibleInterpretation.map((row) => (
                <tr key={row.field}>
                  <td>{row.field}</td>
                  <td>{row.value}</td>
                  <td>{row.why_it_matters}</td>
                </tr>
              ))}
              {!visibleInterpretation.length && <tr><td colSpan={3}>loading live route interpretation</td></tr>}
            </tbody>
          </table>
        </div>
      </section>

      <section className="block">
        <div className="label">SECTION .INPUT</div>
        <textarea value={task} onChange={(event) => setTask(event.target.value)} spellCheck={false} />
        <div className="row">
          <span>PRESET</span>
          {PRESETS.map((preset, index) => (
            <button
              key={preset.label}
              className={task === preset.task ? "active" : ""}
              title={preset.task}
              onClick={() => runSimulation(preset.task, preset.mode)}
            >
              {preset.label}
            </button>
          ))}
          <span>CUSTOM_PROMPT={task === response?.task ? "SYNC" : "DIRTY"}</span>
          <span>WARM_CACHE={preload.loaded}/{preload.total}:{preload.active}</span>
        </div>
        <div className="row mode-row">
          <span>MODE</span>
          {supportedModes.map((option) => (
            <button key={option} className={selectedMode === option ? "active" : ""} onClick={() => runSimulation(undefined, option)}>
              {option.toUpperCase()}
            </button>
          ))}
          <span>STEPS</span>
          <input type="number" min={1} max={256} value={steps} onChange={(event) => setSteps(Number(event.target.value))} />
          <button onClick={() => runSimulation()}>INT 0x80 RUN</button>
          <button onClick={() => setRunning((value) => !value)} disabled={!response?.frames?.length}>
            {running ? "HLT" : "JMP PLAY"}
          </button>
        </div>
        {trainingNotice && <pre className="notice">TRAINING_NOTICE {trainingNotice}</pre>}
        {(response?.warning || error) && <pre className="err">WARN {response?.warning ?? "NONE"}{"\n"}ERR  {error || "NONE"}</pre>}
      </section>

      <section className="block explainer-block">
        <div className="label">SECTION .ANIMATED_EXPLAINER</div>
        <textarea value={explainerPrompt} onChange={(event) => setExplainerPrompt(event.target.value)} spellCheck={false} />
        <div className="row">
          <button onClick={() => runExplainer()} disabled={explainerLoading || !explainerPrompt.trim()}>
            {explainerLoading ? "RENDERING" : "GENERATE"}
          </button>
          <button onClick={() => runExplainer(task)} disabled={explainerLoading || !task.trim()}>USE CURRENT TASK</button>
          <span>JOB {explainerJobId || "IDLE"}</span>
          <span>STATUS {explainerState?.status ?? "IDLE"}</span>
          <span>PROVIDER {explainerState?.provider ?? "PENDING"}</span>
          <span>ATTEMPTS {cell(explainerState?.attempts ?? 0)}</span>
        </div>
        {explainerVideoSrc && (
          <div className="explainer-output">
            <video className="explainer-video" controls src={explainerVideoSrc} />
            <div className="explainer-meta">
              <table>
                <tbody>
                  <tr><th>route</th><td>{cell(explainerState?.route?.recommended_dataset)}</td></tr>
                  <tr><th>reason</th><td>{cell(explainerState?.route?.reason)}</td></tr>
                  <tr><th>label</th><td>{cell(explainerState?.label)}</td></tr>
                  <tr><th>explanation</th><td>{cell(explainerState?.explanation)}</td></tr>
                  <tr><th>brief title</th><td>{cell(authoringBrief?.title)}</td></tr>
                  <tr><th>visual goal</th><td>{cell(authoringBrief?.visual_goal)}</td></tr>
                  <tr><th>anchor dataset</th><td>{cell(authoringBrief?.anchor_dataset)}</td></tr>
                </tbody>
              </table>
              <div className="panel-note">The LLM brief and the exact Manim prompt are visible below.</div>
            </div>
          </div>
        )}
        {authoringBrief?.scene_beats?.length ? (
          <div className="explainer-grid">
            <div className="block">
              <div className="label">SECTION .LLM_BRIEF</div>
              <table>
                <tbody>
                  <tr><th>title</th><td>{cell(authoringBrief?.title)}</td></tr>
                  <tr><th>anchor_dataset</th><td>{cell(authoringBrief?.anchor_dataset)}</td></tr>
                  <tr><th>route_reason</th><td>{cell(authoringBrief?.route_reason)}</td></tr>
                  <tr><th>visual_goal</th><td>{cell(authoringBrief?.visual_goal)}</td></tr>
                  <tr><th>camera</th><td>{Array.isArray(authoringBrief?.camera) ? authoringBrief?.camera.join(" | ") : "unknown"}</td></tr>
                  <tr><th>palette</th><td>{Array.isArray(authoringBrief?.palette) ? authoringBrief?.palette.join(" | ") : "unknown"}</td></tr>
                </tbody>
              </table>
            </div>
            <div className="block">
              <div className="label">SECTION .LLM_BEATS</div>
              <table>
                <tbody>
                  {(authoringBrief?.scene_beats ?? []).map((beat, index) => (
                    <tr key={index}><th>{index + 1}</th><td>{beat}</td></tr>
                  ))}
                  {(authoringBrief?.safety_notes ?? []).map((note, index) => (
                    <tr key={"note-" + index}><th>safety {index + 1}</th><td>{note}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : null}
        {manimPrompt && (
          <div className="block prompt-block">
            <div className="label">SECTION .MANIM_PROMPT</div>
            <pre className="prompt-copy">{manimPrompt}</pre>
            {promptLines.length > 1 && <pre className="prompt-copy muted">{promptLines.join("\n")}</pre>}
          </div>
        )}
        {explainerState?.status === "done_fallback" && <pre className="notice">LIVE_GENERATION_FALLBACK prepared animation served</pre>}
        {(explainerError || explainerState?.reason) && <pre className="err">EXPLAINER_WARN {explainerError || explainerState?.reason}</pre>}
      </section>
      <nav className="tabs">
        {['RUN', 'EXPLAINER', 'INTERPRET', 'DATA', 'MODEL', 'FRAMES', 'RAW'].map((tab) => (
          <button key={tab} className={activeTab === tab ? "active" : ""} onClick={() => setActiveTab(tab)}>
            .{tab}
          </button>
        ))}
      </nav>
      {activeTab === "RUN" && (
        <section className="grid">
          <div className="block">
            <div className="label">SECTION .ROUTE_TABLE</div>
            <table>
              <tbody>
                <tr><th>requested_environment</th><td>{cell(response?.meta?.requested_environment ?? response?.route?.requested_environment)}</td></tr>
                <tr><th>route_reason</th><td>{cell(response?.meta?.route_reason ?? response?.route?.reason)}</td></tr>
                <tr><th>recommended_dataset</th><td>{routedDataset}</td></tr>
                <tr><th>loaded_dataset</th><td>{loadedDataset}</td></tr>
                <tr><th>trained_for_request</th><td>{cell(trainedForRequest ?? "idle")}</td></tr>
                <tr><th>mode</th><td>{selectedMode}</td></tr>
                <tr><th>supported_modes</th><td>{supportedModes.join(", ")}</td></tr>
                <tr><th>prediction</th><td>{cell(response?.meta?.prediction_method)}</td></tr>
                <tr><th>horizon</th><td>{cell(response?.meta?.prediction_horizon)}</td></tr>
              </tbody>
            </table>
          </div>
          <div className="block">
            <div className="label">SECTION .LLM_LAYER</div>
            <table>
              <tbody>
                <tr><th>status</th><td>{cell(response?.llm_explanation?.status ?? modelInfo?.llm_explanation?.status)}</td></tr>
                <tr><th>provider</th><td>{cell(response?.llm_explanation?.provider ?? modelInfo?.llm_explanation?.provider)}</td></tr>
                <tr><th>reason</th><td>{cell(response?.llm_explanation?.reason ?? modelInfo?.llm_explanation?.reason ?? modelInfo?.llm_explanation?.setup)}</td></tr>
                <tr><th>text</th><td>{cell(response?.llm_explanation?.text)}</td></tr>
              </tbody>
            </table>
          </div>
        </section>
      )}

      {activeTab === "EXPLAINER" && (
        <section className="grid">
          <div className="block">
            <div className="label">SECTION .EXPLAINER_JOB</div>
            <table>
              <tbody>
                <tr><th>job_id</th><td>{explainerJobId || "idle"}</td></tr>
                <tr><th>status</th><td>{explainerState?.status ?? "idle"}</td></tr>
                <tr><th>provider</th><td>{explainerState?.provider ?? "pending"}</td></tr>
                <tr><th>url</th><td>{explainerState?.url ?? "pending"}</td></tr>
                <tr><th>attempts</th><td>{cell(explainerState?.attempts ?? 0)}</td></tr>
                <tr><th>warning</th><td>{cell(explainerError || explainerState?.reason)}</td></tr>
              </tbody>
            </table>
          </div>
          <div className="block">
            <div className="label">SECTION .EXPLAINER_ROUTE</div>
            <RawBlock value={explainerState ? { route: explainerState.route, label: explainerState.label, explanation: explainerState.explanation } : null} />
          </div>
        </section>
      )}
      {activeTab === "INTERPRET" && (
        <section className="block">
          <div className="label">SECTION .WHAT_AM_I_SEEING</div>
          <table>
            <thead><tr><th>field</th><th>value</th><th>why useful / limit</th></tr></thead>
            <tbody>
              {(response?.interpretation ?? []).map((row) => (
                <tr key={row.field}>
                  <td>{row.field}</td>
                  <td>{row.value}</td>
                  <td>{row.why_it_matters}</td>
                </tr>
              ))}
              {!response?.interpretation?.length && (
                <tr><td>idle</td><td>run a simulation</td><td>the interpretation table is built from live route + checkpoint metadata</td></tr>
              )}
            </tbody>
          </table>
        </section>
      )}

      {activeTab === "DATA" && (
        <section className="block">
          <div className="label">SECTION .THE_WELL_DATASETS</div>
          <table className="dense">
            <thead>
              <tr>
                <th>dataset</th><th>domain</th><th>cs/res</th><th>steps/traj/GB</th><th>fields</th><th>useful</th><th>not useful</th><th>local</th>
              </tr>
            </thead>
            <tbody>
              {catalog.map((row) => (
                <tr key={row.name}>
                  <td><a href={row.source_url} target="_blank">{row.name}</a><br />{row.family}</td>
                  <td>{row.domain}</td>
                  <td>{row.coordinate_system}<br />{row.resolution}</td>
                  <td>{row.n_steps} / {row.n_traj} / {row.size_gb}</td>
                  <td>{row.fields}</td>
                  <td>{row.useful_for}</td>
                  <td>{row.not_useful_for}</td>
                  <td>{localDatasetStatus(row.name)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <pre className="sources">{`sources:
${cell(modelInfo?.catalog_summary?.source_home)}
${cell(modelInfo?.catalog_summary?.source_overview)}
${cell(modelInfo?.catalog_summary?.source_huggingface)}`}</pre>
        </section>
      )}

      {activeTab === "MODEL" && (
        <section className="grid">
          <div className="block">
            <div className="label">SECTION .MODEL_TABLE</div>
            <table>
              <tbody>
                <tr><th>checkpoint</th><td>{modelInfo?.checkpoint_loaded ? "loaded" : "missing"}</td></tr>
                <tr><th>artifact</th><td>{modelInfo?.checkpoint_path ?? "loading"}</td></tr>
                <tr><th>dataset</th><td>{cell(modelInfo?.meta?.dataset)}</td></tr>
                <tr><th>field</th><td>{cell(modelInfo?.meta?.field)}</td></tr>
                <tr><th>frame_size</th><td>{cell(modelInfo?.meta?.frame_size)}</td></tr>
                <tr><th>model_width</th><td>{cell(modelInfo?.meta?.model_width)}</td></tr>
                <tr><th>epochs</th><td>{cell(budget?.epochs)}</td></tr>
                <tr><th>train_batches/epoch</th><td>{cell(budget?.max_train_batches_per_epoch)}</td></tr>
                <tr><th>valid_batches/epoch</th><td>{cell(budget?.max_validation_batches_per_epoch)}</td></tr>
                <tr><th>seed_bank</th><td>{cell(budget?.seed_bank_size)}</td></tr>
                <tr><th>train_loss</th><td>{cell(modelInfo?.meta?.train_loss)}</td></tr>
                <tr><th>val_loss</th><td>{cell(modelInfo?.meta?.val_loss ?? modelInfo?.training_report?.best_val_loss)}</td></tr>
              </tbody>
            </table>
          </div>
          <div className="block">
            <div className="label">SECTION .TRAINING_LOG</div>
            <table>
              <thead><tr><th>epoch</th><th>train_mse</th><th>valid_mse</th></tr></thead>
              <tbody>
                {history.map((row, index) => (
                  <tr key={index}>
                    <td>{cell(row.epoch)}</td>
                    <td>{cell(row.train_loss)}</td>
                    <td>{cell(row.val_loss)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="block">
            <div className="label">SECTION .MODE_REGISTRY</div>
            <table>
              <thead><tr><th>dataset</th><th>local</th><th>modes</th></tr></thead>
              <tbody>
                {simulationModes.map((row) => (
                  <tr key={String(row.dataset)}>
                    <td>{cell(row.dataset)}</td>
                    <td>{row.checkpoint_exists === true ? "checkpoint" : `fallback=${cell(row.fallback)}`}</td>
                    <td>{Array.isArray(row.supported_modes) ? row.supported_modes.map(String).join(", ") : "unknown"}</td>
                  </tr>
                ))}
                {!simulationModes.length && <tr><td>loading</td><td colSpan={2}>mode registry unavailable</td></tr>}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {activeTab === "FRAMES" && (
        <section className="block">
          <div className="label">SECTION .FRAME_STATS</div>
          <table>
            <thead><tr><th>frame</th><th>min</th><th>max</th><th>mean</th><th>std</th><th>gradient_energy</th></tr></thead>
            <tbody>
              {frameStats.map((row) => (
                <tr key={row.frame}>
                  <td>{row.frame}</td>
                  <td>{cell(row.min)}</td>
                  <td>{cell(row.max)}</td>
                  <td>{cell(row.mean)}</td>
                  <td>{cell(row.std)}</td>
                  <td>{cell(row.gradient_energy)}</td>
                </tr>
              ))}
              {!frameStats.length && <tr><td>idle</td><td colSpan={5}>run /simulate to fill the prediction statistics table</td></tr>}
            </tbody>
          </table>
        </section>
      )}

      {activeTab === "RAW" && (
        <section className="grid">
          <div className="block">
            <div className="label">SECTION .SIM_RAW_NO_FRAMES</div>
            <RawBlock value={response ? { task: response.task, meta: response.meta, route: response.route, warning: response.warning } : null} />
          </div>
          <div className="block">
            <div className="label">SECTION .MODEL_RAW</div>
            <RawBlock value={modelInfo ? { meta: modelInfo.meta, catalog_summary: modelInfo.catalog_summary, datasets_trained: modelInfo.datasets_trained } : null} />
          </div>
        </section>
      )}
    </main>
  );
}

