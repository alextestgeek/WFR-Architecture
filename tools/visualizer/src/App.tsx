import { useState, useEffect, useRef, useCallback } from 'react'
import { CoreScene3D } from './CoreScene3D.tsx'
import { Play, RotateCcw, Zap, FlaskConical, Sun, Moon } from 'lucide-react'
import { motion } from 'framer-motion'

const LIVE_WS =
  (import.meta as { env?: { VITE_WFR_LIVE_WS?: string } }).env?.VITE_WFR_LIVE_WS ??
  'ws://127.0.0.1:8765/ws'

// ─────────────────────────────────────────────────────
// WFR CORE — Real implementation (mirrors wfr_core.py)
// ─────────────────────────────────────────────────────

function wavePhaseEncode(
  positions: number[], numPhases: number, numFractalLevels: number,
  alpha: number[], beta: number[], gamma: number[], baseFreqs: number[]
): number[][] {
  return positions.map(i =>
    baseFreqs.map((f, m) => {
      let phase = 2 * Math.PI * f * i
      for (let l = 0; l < numFractalLevels; l++) {
        phase += alpha[l] * Math.sin(2 * Math.PI * beta[l] * Math.log2(i + 2) + gamma[l])
      }
      return ((phase % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)
    })
  )
}

function resonanceFunction(z: number, frequency: number, threshold: number, decay: number): number {
  if (Math.abs(z) <= threshold) return 0
  return Math.sin(2 * Math.PI * frequency * z) * Math.exp(-decay * Math.abs(z - threshold))
}

function computeResonanceLayer(
  phases: number[][], weights: number[], frequency: number, threshold: number, decay: number, spikeThreshold: number
) {
  const resonances = phases.map(pv => {
    const combined = pv.reduce((s, p, j) => s + p * weights[j], 0)
    return resonanceFunction(combined, frequency, threshold, decay)
  })
  const spikes = resonances.map(r => Math.abs(r) > spikeThreshold)
  return { resonances, spikes }
}

function computeResonanceConfidence(phases: number[][]): number {
  if (phases.length === 0) return 0
  const nPh = phases[0].length
  let total = 0
  for (let m = 0; m < nPh; m++) {
    let re = 0, im = 0
    for (const pv of phases) { re += Math.cos(pv[m]); im += Math.sin(pv[m]) }
    re /= phases.length; im /= phases.length
    total += Math.sqrt(re * re + im * im)
  }
  return total / nPh
}

// ─────────────────────────────────────────────────────
// Model: created once from seed, then reused across contexts
// Mirrors: WFRNetwork.__init__ in wfr_core.py
// ─────────────────────────────────────────────────────

interface WFRModel {
  numPhases: number
  numFractalLevels: number
  baseFreqs: number[]
  alpha: number[]
  beta: number[]
  gamma: number[]
  layers: { weights: number[]; frequency: number; threshold: number; decay: number }[]
  spikeThreshold: number
}

function createModel(numPhases: number, numLevels: number, seed: number): WFRModel {
  let _s = (seed | 0) || 1
  const rand01 = () => { _s ^= _s << 13; _s ^= _s >> 17; _s ^= _s << 5; return ((_s >>> 0) % 100000) / 100000 }
  const randn = () => { const u1 = rand01() || 0.0001, u2 = rand01(); return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) }

  const numFractalLevels = 6
  return {
    numPhases,
    numFractalLevels,
    baseFreqs: Array.from({ length: numPhases }, (_, m) => Math.pow(2, m)),
    alpha: Array.from({ length: numFractalLevels }, () => randn() * 0.1),
    beta: Array.from({ length: numFractalLevels }, () => randn() * 0.5 + 1.0),
    gamma: Array.from({ length: numFractalLevels }, () => randn() * Math.PI),
    layers: Array.from({ length: numLevels }, (_, i) => ({
      weights: Array.from({ length: numPhases }, () => randn() * 0.1),
      frequency: 1.0 + i * 0.5,
      threshold: 0.2 + i * 0.05,
      decay: 2.0,
    })),
    spikeThreshold: 0.4,
  }
}

// ─────────────────────────────────────────────────────

interface LayerResult {
  level: number; spikeRate: number; silentPct: number
  avgAmplitude: number; resonances: number[]; spikes: boolean[]
}

interface WFRResult {
  phases: number[][]; layers: LayerResult[]; rc: number
  standingWave: number[]; contextLength: number
}

function runInference(model: WFRModel, contextLength: number): WFRResult {
  const step = Math.max(1, Math.floor(contextLength / 512))
  const positions = Array.from({ length: Math.min(contextLength, 512) }, (_, i) => i * step)
  const phases = wavePhaseEncode(positions, model.numPhases, model.numFractalLevels, model.alpha, model.beta, model.gamma, model.baseFreqs)

  const layers: LayerResult[] = model.layers.map((lp, i) => {
    const { resonances, spikes } = computeResonanceLayer(phases, lp.weights, lp.frequency, lp.threshold, lp.decay, model.spikeThreshold)
    const sr = spikes.filter(Boolean).length / spikes.length
    return {
      level: i, spikeRate: sr, silentPct: (1 - sr) * 100,
      avgAmplitude: resonances.reduce((s, r) => s + Math.abs(r), 0) / resonances.length,
      resonances, spikes,
    }
  })

  const rc = computeResonanceConfidence(phases)
  const standingWave = positions.map((_, idx) =>
    layers.reduce((sum, l) => sum + l.resonances[idx], 0) / layers.length
  )
  return { phases, layers, rc, standingWave, contextLength }
}

// ─────────────────────────────────────────────────────
// THEME
// ─────────────────────────────────────────────────────

const themes = {
  dark: {
    bg: '#09090b', surface: '#18181b', border: 'rgba(255,255,255,0.08)',
    text: '#fafafa', textMuted: '#71717a', textDim: '#3f3f46',
    canvasBg: '#000', gridLine: 'rgba(255,255,255,0.04)', zeroLine: 'rgba(255,255,255,0.1)',
    cardBg: 'rgba(39,39,42,0.5)', btnBg: '#fff', btnText: '#000',
    btnOutline: 'rgba(255,255,255,0.15)', accent: '#22d3ee',
  },
  light: {
    bg: '#f8f8fa', surface: '#ffffff', border: 'rgba(0,0,0,0.08)',
    text: '#18181b', textMuted: '#71717a', textDim: '#a1a1aa',
    canvasBg: '#fff', gridLine: 'rgba(0,0,0,0.05)', zeroLine: 'rgba(0,0,0,0.12)',
    cardBg: 'rgba(244,244,245,0.8)', btnBg: '#18181b', btnText: '#fff',
    btnOutline: 'rgba(0,0,0,0.12)', accent: '#0891b2',
  },
} as const

type ThemeKey = keyof typeof themes

type ThemePalette = (typeof themes)[ThemeKey]

const LAYER_COLORS = ['#818cf8', '#34d399', '#fbbf24', '#f87171', '#a78bfa', '#2dd4bf', '#fb923c', '#e879f9']
const CTX_LENGTHS = [512, 2048, 8192]

// ─────────────────────────────────────────────────────
// Canvas drawing helpers
// ─────────────────────────────────────────────────────

function initCanvas(canvas: HTMLCanvasElement) {
  const dpr = window.devicePixelRatio || 1
  canvas.width = canvas.clientWidth * dpr
  canvas.height = canvas.clientHeight * dpr
  const ctx = canvas.getContext('2d')!
  ctx.scale(dpr, dpr)
  return { ctx, w: canvas.clientWidth, h: canvas.clientHeight, dpr }
}

function drawStandingWave(canvas: HTMLCanvasElement, result: WFRResult, t: ThemePalette) {
  const { ctx, w, h } = initCanvas(canvas)
  ctx.fillStyle = t.canvasBg; ctx.fillRect(0, 0, w, h)

  ctx.strokeStyle = t.gridLine; ctx.lineWidth = 1
  for (let x = 0; x < w; x += 40) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke() }
  for (let y = 0; y < h; y += 40) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke() }
  ctx.strokeStyle = t.zeroLine; ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke()

  const sw = result.standingWave
  if (!sw.length) return
  const maxAbs = Math.max(...sw.map(Math.abs), 0.001)

  result.layers.forEach((layer, li) => {
    ctx.strokeStyle = LAYER_COLORS[li % LAYER_COLORS.length]; ctx.globalAlpha = 0.35; ctx.lineWidth = 1
    ctx.beginPath()
    layer.resonances.forEach((v, i) => {
      const x = (i / (layer.resonances.length - 1)) * w
      const y = h / 2 - (v / maxAbs) * (h * 0.36)
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
    })
    ctx.stroke()
  })
  ctx.globalAlpha = 1

  ctx.strokeStyle = t.accent; ctx.lineWidth = 2.5; ctx.shadowColor = t.accent; ctx.shadowBlur = 16
  ctx.beginPath()
  sw.forEach((v, i) => {
    const x = (i / (sw.length - 1)) * w, y = h / 2 - (v / maxAbs) * (h * 0.36)
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
  })
  ctx.stroke(); ctx.shadowBlur = 0

  if (result.layers[0]) {
    ctx.fillStyle = '#f472b6'; ctx.shadowColor = '#f472b6'; ctx.shadowBlur = 10
    result.layers[0].spikes.forEach((s, i) => {
      if (!s) return
      const x = (i / (result.layers[0].spikes.length - 1)) * w
      const y = h / 2 - (sw[i] / maxAbs) * (h * 0.36)
      ctx.beginPath(); ctx.arc(x, y, 2.5, 0, Math.PI * 2); ctx.fill()
    })
    ctx.shadowBlur = 0
  }
}

function drawPhaseMap(canvas: HTMLCanvasElement, result: WFRResult, themeKey: ThemeKey) {
  const { ctx: c, dpr } = initCanvas(canvas)
  const ph = result.phases, nPos = ph.length, nPh = ph[0]?.length || 0
  if (!nPos || !nPh) return

  const twilight = (v: number): [number, number, number] => {
    const t2 = v * 2 * Math.PI
    const r = Math.round(78 + 100 * Math.sin(t2) ** 2 + 40 * Math.sin(t2 * 0.5) ** 2)
    const g = Math.round(60 + 60 * Math.sin(t2 + 2.1) ** 2 + 60 * Math.cos(t2 * 0.5) ** 2)
    const b = Math.round(100 + 110 * Math.cos(t2) ** 2 + 30 * Math.sin(t2 * 0.5 + 1) ** 2)
    return [Math.min(255, r), Math.min(255, g), Math.min(255, b)]
  }

  const imgData = c.createImageData(canvas.width, canvas.height)
  const cW = canvas.width / nPos, cH = canvas.height / nPh
  for (let p = 0; p < nPh; p++) {
    const y0 = Math.floor(p * cH), y1 = Math.floor((p + 1) * cH)
    for (let i = 0; i < nPos; i++) {
      const x0 = Math.floor(i * cW), x1 = Math.floor((i + 1) * cW)
      const v = ph[i][p] / (2 * Math.PI)
      const [r, g, b] = twilight(v)
      for (let py = y0; py < y1; py++) for (let px = x0; px < x1; px++) {
        const idx = (py * canvas.width + px) * 4
        imgData.data[idx] = r; imgData.data[idx + 1] = g; imgData.data[idx + 2] = b; imgData.data[idx + 3] = 255
      }
    }
  }
  c.putImageData(imgData, 0, 0)

  c.fillStyle = themeKey === 'dark' ? 'rgba(255,255,255,0.5)' : 'rgba(0,0,0,0.5)'
  c.font = '9px JetBrains Mono, monospace'; c.textAlign = 'right'
  const labelStep = nPh <= 8 ? 1 : nPh <= 16 ? 2 : 4
  for (let p = 0; p < nPh; p += labelStep) c.fillText(String(p), 16 / dpr, ((p + 0.6) * cH) / dpr)
}

function drawActivityBars(canvas: HTMLCanvasElement, result: WFRResult, t: ThemePalette, themeKey: ThemeKey) {
  const { ctx, w, h } = initCanvas(canvas)
  ctx.fillStyle = t.canvasBg; ctx.fillRect(0, 0, w, h)

  const layers = result.layers, n = layers.length
  const pad = 32, gap = 8, barW = (w - pad * 2 - gap * (n - 1)) / n

  ctx.fillStyle = t.textDim; ctx.font = '10px JetBrains Mono, monospace'; ctx.textAlign = 'right'
  for (let pct = 0; pct <= 100; pct += 25) {
    const y = pad + (1 - pct / 100) * (h - pad * 2)
    ctx.fillText(`${pct}%`, pad - 6, y + 3)
    ctx.strokeStyle = t.gridLine; ctx.lineWidth = 1
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(w - pad, y); ctx.stroke()
  }

  layers.forEach((layer, i) => {
    const x = pad + i * (barW + gap), totalH = h - pad * 2
    const activeH = layer.spikeRate * totalH, silentH = totalH - activeH
    ctx.fillStyle = themeKey === 'dark' ? '#27272a' : '#d4d4d8'
    ctx.fillRect(x, pad, barW, silentH)
    ctx.fillStyle = '#f59e0b'
    ctx.fillRect(x, pad + silentH, barW, activeH)
    ctx.fillStyle = t.textMuted; ctx.font = '10px JetBrains Mono, monospace'; ctx.textAlign = 'center'
    ctx.fillText(`L${i}`, x + barW / 2, h - 8)
    ctx.fillStyle = themeKey === 'dark' ? '#fff' : '#18181b'
    ctx.font = 'bold 11px JetBrains Mono, monospace'
    const labelY = activeH > 16 ? pad + silentH + activeH / 2 + 4 : pad + silentH - 6
    ctx.fillText(`${(layer.spikeRate * 100).toFixed(1)}%`, x + barW / 2, labelY)
  })

  ctx.fillStyle = t.textMuted; ctx.font = '10px JetBrains Mono, monospace'; ctx.textAlign = 'left'
  ctx.fillText('ACTIVE (amber) vs SILENT (grey)', pad, 16)
}

function drawResonanceAmplitudes(canvas: HTMLCanvasElement, result: WFRResult, t: ThemePalette) {
  const { ctx, w, h } = initCanvas(canvas)
  const pad = { top: 28, right: 16, bottom: 28, left: 44 }
  const plotW = w - pad.left - pad.right, plotH = h - pad.top - pad.bottom
  ctx.fillStyle = t.canvasBg; ctx.fillRect(0, 0, w, h)

  ctx.strokeStyle = t.gridLine; ctx.lineWidth = 1
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (i / 4) * plotH
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke()
  }
  const zeroY = pad.top + plotH / 2
  ctx.strokeStyle = t.zeroLine; ctx.lineWidth = 1
  ctx.beginPath(); ctx.moveTo(pad.left, zeroY); ctx.lineTo(w - pad.right, zeroY); ctx.stroke()

  let globalMax = 0.001
  for (const layer of result.layers) for (const v of layer.resonances) if (Math.abs(v) > globalMax) globalMax = Math.abs(v)

  result.layers.forEach((layer, li) => {
    const res = layer.resonances, n = res.length, step = Math.max(1, Math.floor(n / 500))
    ctx.strokeStyle = LAYER_COLORS[li % LAYER_COLORS.length]; ctx.lineWidth = 1.4; ctx.globalAlpha = 0.85
    ctx.beginPath()
    let first = true
    for (let i = 0; i < n; i += step) {
      const x = pad.left + (i / (n - 1)) * plotW, y = zeroY - (res[i] / globalMax) * (plotH * 0.45)
      first ? ctx.moveTo(x, y) : ctx.lineTo(x, y); first = false
    }
    ctx.stroke()
  })
  ctx.globalAlpha = 1

  ctx.fillStyle = t.textDim; ctx.font = '9px JetBrains Mono, monospace'; ctx.textAlign = 'right'
  for (let i = 0; i <= 4; i++) {
    const val = (1 - i / 2) * globalMax
    ctx.fillText(val.toFixed(2), pad.left - 4, pad.top + (i / 4) * plotH + 3)
  }
  ctx.textAlign = 'center'
  ctx.fillText('Position', pad.left + plotW / 2, h - 4)

  ctx.font = '9px JetBrains Mono, monospace'
  result.layers.forEach((_, li) => {
    const lx = pad.left + 8 + li * 56
    ctx.fillStyle = LAYER_COLORS[li % LAYER_COLORS.length]
    ctx.fillRect(lx, pad.top + 8, 10, 3)
    ctx.fillText(`L${li}`, lx + 14, pad.top + 12)
  })
}

// ─────────────────────────────────────────────────────
// Single context panel (4 charts like Python subplot 2x2)
// ─────────────────────────────────────────────────────

function ContextPanel({ result, theme, t }: { result: WFRResult; theme: ThemeKey; t: ThemePalette }) {
  const waveRef = useRef<HTMLCanvasElement>(null)
  const phaseRef = useRef<HTMLCanvasElement>(null)
  const activityRef = useRef<HTMLCanvasElement>(null)
  const ampRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    if (waveRef.current) drawStandingWave(waveRef.current, result, t)
    if (phaseRef.current) drawPhaseMap(phaseRef.current, result, theme)
    if (activityRef.current) drawActivityBars(activityRef.current, result, t, theme)
    if (ampRef.current) drawResonanceAmplitudes(ampRef.current, result, t)
  }, [result, theme])

  return (
    <div className="rounded-2xl overflow-hidden" style={{ border: `1px solid ${t.border}` }}>
      <div className="px-5 py-3 text-[12px] font-mono flex items-center justify-between"
        style={{ background: t.cardBg, borderBottom: `1px solid ${t.border}` }}>
        <span className="font-semibold">context = {result.contextLength.toLocaleString()}</span>
        <span style={{ color: t.accent }}>RC = {result.rc.toFixed(4)}</span>
      </div>

      <div className="grid grid-cols-2">
        {/* Standing Wave */}
        <div style={{ borderRight: `1px solid ${t.border}`, borderBottom: `1px solid ${t.border}` }}>
          <div className="px-3 py-1.5 text-[10px] font-mono" style={{ color: t.textMuted, background: t.canvasBg }}>
            Standing Wave (средний резонанс по слоям)
          </div>
          <canvas ref={waveRef} className="w-full" style={{ height: 200 }} />
        </div>
        {/* Phase Map */}
        <div style={{ borderBottom: `1px solid ${t.border}` }}>
          <div className="px-3 py-1.5 text-[10px] font-mono" style={{ color: t.textMuted, background: t.canvasBg }}>
            Phase Map (первые {result.phases[0]?.length ?? 0} фаз)
          </div>
          <canvas ref={phaseRef} className="w-full" style={{ height: 200 }} />
        </div>
        {/* Activity bars */}
        <div style={{ borderRight: `1px solid ${t.border}` }}>
          <div className="px-3 py-1.5 text-[10px] font-mono" style={{ color: t.textMuted, background: t.canvasBg }}>
            Event-Driven Activity по слоям
          </div>
          <canvas ref={activityRef} className="w-full" style={{ height: 200 }} />
        </div>
        {/* Resonance Amplitudes */}
        <div>
          <div className="px-3 py-1.5 text-[10px] font-mono" style={{ color: t.textMuted, background: t.canvasBg }}>
            Resonance Amplitudes по слоям
          </div>
          <canvas ref={ampRef} className="w-full" style={{ height: 200 }} />
        </div>
      </div>

      {/* Layer stats row */}
      <div className="grid gap-0" style={{ gridTemplateColumns: `repeat(${result.layers.length}, 1fr)`, borderTop: `1px solid ${t.border}` }}>
        {result.layers.map(layer => (
          <div key={layer.level} className="px-3 py-2.5 text-center" style={{ borderRight: `1px solid ${t.border}` }}>
            <div className="text-[9px] font-mono" style={{ color: t.textDim }}>Layer {layer.level}</div>
            <div className="text-[11px] font-mono mt-0.5">
              spk=<span style={{ color: '#f59e0b' }}>{(layer.spikeRate * 100).toFixed(1)}%</span>
              {' '}silent=<span style={{ color: '#06b6d4' }}>{layer.silentPct.toFixed(1)}%</span>
              {' '}amp=<span style={{ color: '#a78bfa' }}>{layer.avgAmplitude.toFixed(4)}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────
// APP
// ─────────────────────────────────────────────────────

const App = () => {
  const [liveOn, setLiveOn] = useState(false)
  const [liveResult, setLiveResult] = useState<WFRResult | null>(null)
  const [liveStatus, setLiveStatus] = useState<'off' | 'connecting' | 'live' | 'err'>('off')

  const liveCloseOkRef = useRef(true)

  useEffect(() => {
    if (!liveOn) {
      setLiveStatus('off')
      setLiveResult(null)
      return
    }
    setLiveStatus('connecting')
    liveCloseOkRef.current = false
    const ws = new WebSocket(LIVE_WS)
    ws.onopen = () => setLiveStatus('live')
    ws.onmessage = (ev) => {
      try {
        const d = JSON.parse(ev.data as string)
        if (d.type !== 'telemetry') return
        const lr = d.layers.map((L: {
          level: number
          spikeRate: number
          silentPct: number
          avgAmplitude: number
          resonances: number[]
          spikes: boolean[]
        }) => ({
          level: L.level,
          spikeRate: L.spikeRate,
          silentPct: L.silentPct,
          avgAmplitude: L.avgAmplitude,
          resonances: L.resonances,
          spikes: L.spikes,
        }))
        setLiveResult({
          phases: d.phases,
          layers: lr,
          rc: d.rc,
          standingWave: d.standingWave,
          contextLength: d.contextLength,
        })
      } catch {
        setLiveStatus('err')
      }
    }
    ws.onerror = () => setLiveStatus('err')
    ws.onclose = () => {
      if (!liveCloseOkRef.current) {
        setLiveStatus('err')
        setLiveOn(false)
      }
    }
    return () => {
      liveCloseOkRef.current = true
      ws.close()
    }
  }, [liveOn])

  const [theme, setTheme] = useState<ThemeKey>('dark')
  const [numLevels, setNumLevels] = useState(4)
  const [numPhases, setNumPhases] = useState(16)
  const [seed, setSeed] = useState(42)
  const [results, setResults] = useState<WFRResult[]>([])
  const [elapsed, setElapsed] = useState('')

  const t = themes[theme]

  const runTest = useCallback(() => {
    const t0 = performance.now()
    const model = createModel(numPhases, numLevels, seed)
    const res = CTX_LENGTHS.map(ctx => runInference(model, ctx))
    setElapsed((performance.now() - t0).toFixed(1) + 'ms')
    setResults(res)
  }, [numPhases, numLevels, seed])

  useEffect(() => { runTest() }, [])

  const resetAll = () => { setSeed(Math.floor(Math.random() * 10000)); setResults([]) }

  return (
    <div className="min-h-screen transition-colors duration-300" style={{ background: t.bg, color: t.text, fontFamily: "'Outfit', system-ui, sans-serif" }}>
      <div className="max-w-[1800px] mx-auto">

        {/* Header */}
        <header className="flex items-center justify-between px-10 py-5" style={{ borderBottom: `1px solid ${t.border}` }}>
          <div className="flex items-center gap-x-4">
            <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-gradient-to-br from-cyan-400 via-violet-500 to-fuchsia-500">
              <Zap className="h-5 w-5 text-white" />
            </div>
            <div>
              <div className="text-3xl font-semibold tracking-tighter">WFR Smoke Test</div>
              <div className="text-[10px] font-mono tracking-[2px]" style={{ color: t.textDim }}>WAVE • FRACTAL • RESONANT — contexts: {CTX_LENGTHS.join(', ')}</div>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <button onClick={() => setTheme(th => th === 'dark' ? 'light' : 'dark')}
              className="p-2.5 rounded-xl transition-colors" style={{ background: t.cardBg, border: `1px solid ${t.border}` }}>
              {theme === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </button>
            <button
              type="button"
              onClick={() => setLiveOn((v) => !v)}
              className="px-3 py-2 rounded-xl text-[11px] font-mono transition-colors"
              style={{
                background: liveOn ? 'rgba(34,211,238,0.15)' : t.cardBg,
                border: `1px solid ${liveOn ? 'rgba(34,211,238,0.5)' : t.border}`,
                color: liveOn ? t.accent : t.textMuted,
              }}
            >
              {liveOn ? 'LIVE: Py core' : 'Live off'}
            </button>
            {liveOn && (
              <span className="text-[10px] font-mono" style={{ color: t.textDim }}>
                {liveStatus === 'live' ? 'stream' : liveStatus}
              </span>
            )}
            <div className="flex items-center gap-2 px-4 py-2 rounded-full" style={{ background: 'rgba(16,185,129,0.1)', border: '1px solid rgba(16,185,129,0.2)' }}>
              <FlaskConical className="h-3.5 w-3.5 text-emerald-500" />
              <span className="text-[11px] font-mono text-emerald-500 tracking-wide">REAL WFR ENGINE</span>
            </div>
            {elapsed && <div className="text-[11px] font-mono" style={{ color: t.textDim }}>{elapsed}</div>}
          </div>
        </header>

        <div className="flex">
          {/* Sidebar */}
          <div className="w-[280px] p-6 flex flex-col gap-6" style={{ borderRight: `1px solid ${t.border}` }}>
            <div>
              <div className="uppercase text-[10px] tracking-[2px] mb-4" style={{ color: t.textDim }}>MODEL PARAMETERS</div>
              <div className="space-y-5">
                {[
                  { label: 'RESONANCE LAYERS', value: String(numLevels), min: 2, max: 8, step: 1, set: setNumLevels, v: numLevels },
                  { label: 'PHASE DIMENSIONS', value: String(numPhases), min: 4, max: 24, step: 1, set: setNumPhases, v: numPhases },
                  { label: 'SEED', value: String(seed), min: 1, max: 9999, step: 1, set: setSeed, v: seed },
                ].map(p => (
                  <div key={p.label}>
                    <div className="flex justify-between text-[11px] mb-2">
                      <span style={{ color: t.textMuted }}>{p.label}</span>
                      <span className="font-mono" style={{ color: t.text }}>{p.value}</span>
                    </div>
                    <input type="range" min={p.min} max={p.max} step={p.step} value={p.v}
                      onChange={e => p.set(+e.target.value)} className="w-full accent-cyan-500" />
                  </div>
                ))}
              </div>
            </div>

            <div className="text-[10px] font-mono p-3 rounded-xl" style={{ background: t.cardBg, color: t.textMuted }}>
              Одна модель → {CTX_LENGTHS.length} контекста<br />
              Как в Python smoke test
            </div>

            {results.length > 0 && (
              <div className="rounded-2xl p-4 space-y-2" style={{ background: t.cardBg, border: `1px solid ${t.border}` }}>
                <div className="uppercase text-[10px] tracking-[2px]" style={{ color: t.textDim }}>SUMMARY</div>
                {results.map(r => (
                  <div key={r.contextLength} className="flex justify-between text-[11px] font-mono">
                    <span style={{ color: t.textMuted }}>ctx={r.contextLength}</span>
                    <span style={{ color: r.rc > 0.8 ? '#10b981' : r.rc > 0.5 ? '#f59e0b' : '#ef4444' }}>
                      RC={r.rc.toFixed(4)}
                    </span>
                  </div>
                ))}
              </div>
            )}

            <div className="flex gap-3 mt-auto">
              <button onClick={runTest} className="flex-1 py-3.5 rounded-2xl font-medium flex items-center justify-center gap-2 transition-all active:scale-[0.98]"
                style={{ background: t.btnBg, color: t.btnText }}>
                <Play className="w-4 h-4" /> RUN ALL
              </button>
              <button onClick={resetAll} className="px-4 rounded-2xl transition-colors" style={{ border: `1px solid ${t.btnOutline}` }}>
                <RotateCcw className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Main — 3 context panels */}
          <div className="flex-1 p-6 space-y-6 overflow-y-auto" style={{ maxHeight: 'calc(100vh - 72px)' }}>
            {liveOn && liveResult && (
              <div className="space-y-3">
                <div className="text-[11px] font-mono" style={{ color: t.textMuted }}>
                  Поток с <b>wfr.core.WFRNetwork</b> (синтетические positions), ~8 Hz
                </div>
                <CoreScene3D result={liveResult} theme={{ bg: t.canvasBg, border: t.border, accent: t.accent }} />
                <ContextPanel result={liveResult} theme={theme} t={t} />
              </div>
            )}

            {results.map(r => (
              <motion.div key={r.contextLength}
                initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
                transition={{ delay: CTX_LENGTHS.indexOf(r.contextLength) * 0.1, type: 'spring', stiffness: 200, damping: 24 }}>
                <ContextPanel result={r} theme={theme} t={t} />
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
