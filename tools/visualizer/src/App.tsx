import { useState, useEffect, useRef, useCallback } from 'react'
import { Play, Pause, RotateCcw, Zap } from 'lucide-react'
import * as THREE from 'three'
import { motion, AnimatePresence } from 'framer-motion'

interface LayerStats {
  level: number
  activeSpikes: number
  avgAmplitude: number
}

const WFRAVisualizer = () => {
  const [isRunning, setIsRunning] = useState(false)
  const [time, setTime] = useState(0)
  const [contextLength, setContextLength] = useState(512)
  const [levels, setLevels] = useState(8)
  const [phaseCount, setPhaseCount] = useState(12)
  const [loss, setLoss] = useState(0.087)
  const [spikeRate, setSpikeRate] = useState(0.14)

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const threeRef = useRef<HTMLDivElement>(null)
  const sceneRef = useRef<THREE.Scene | null>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const waveMeshRef = useRef<THREE.Mesh | null>(null)
  const animationFrameRef = useRef<number | null>(null)

  const [layers, setLayers] = useState<LayerStats[]>([])
  const [phases, setPhases] = useState<number[]>([])

  const simulateStep = useCallback((t: number, ctxLen: number) => {
    const newPhases: number[] = []
    const newLayers: LayerStats[] = []

    for (let i = 0; i < 128; i++) {
      let phase = 0
      for (let m = 0; m < phaseCount; m++) {
        const freq = Math.pow(2, m % 6)
        phase += Math.sin(2 * Math.PI * freq * (i / ctxLen) + t * 0.8) * 0.3
      }
      newPhases.push((phase + 1) / 2)
    }

    for (let l = 0; l < levels; l++) {
      const resonance = Math.sin(t * (1.2 + l * 0.3)) * 0.5 + 0.5
      const spikes = Math.floor(18 + Math.random() * 12 * (1 - l / levels))
      
      newLayers.push({
        level: l,
        activeSpikes: spikes,
        avgAmplitude: resonance * (0.6 + Math.random() * 0.4)
      })
    }

    setPhases(newPhases)
    setLayers(newLayers)
    setLoss(0.042 + Math.sin(t * 0.3) * 0.015)
    setSpikeRate(0.08 + Math.random() * 0.12)

    return newPhases
  }, [levels, phaseCount])

  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d', { alpha: true })
    if (!ctx) return

    ctx.clearRect(0, 0, canvas.width, canvas.height)

    ctx.strokeStyle = 'rgba(163, 163, 163, 0.06)'
    ctx.lineWidth = 1
    for (let x = 0; x < canvas.width; x += 32) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke()
    }

    const w = canvas.width
    const h = canvas.height
    const t = time

    ctx.strokeStyle = '#67e8f9'
    ctx.lineWidth = 3.5
    ctx.shadowColor = '#67e8f9'
    ctx.shadowBlur = 25

    ctx.beginPath()
    for (let x = 0; x < w; x += 1.5) {
      const norm = x / w
      let y = h * 0.48
      
      for (let k = 0; k < 4; k++) {
        const p = phases[Math.floor(norm * (phases.length - 1))] || 0.5
        y += Math.sin(norm * 14 + t * 1.6 + k * 1.4) * (p * 32 + 12)
      }
      
      if (x === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    }
    ctx.stroke()

    ctx.fillStyle = '#e879f9'
    for (let i = 0; i < 8; i++) {
      const px = ((w * 0.1) + (t * 42 % (w * 0.8))) + i * 48
      const py = h * 0.5 + Math.sin(t * 2.4 + i * 0.8) * 42
      ctx.save()
      ctx.shadowBlur = 35
      ctx.shadowColor = '#e879f9'
      ctx.beginPath()
      ctx.arc(px % w, py, 4.5, 0, Math.PI * 2)
      ctx.fill()
      ctx.restore()
    }

    ctx.strokeStyle = '#fde047'
    ctx.lineWidth = 2
    layers.slice(0, 6).forEach((layer, idx) => {
      const y = 70 + idx * 38
      const count = Math.min(layer.activeSpikes, 11)
      for (let s = 0; s < count; s++) {
        const x = 120 + (s * 29) + Math.sin(t * 2 + idx) * 9
        ctx.beginPath()
        ctx.moveTo(x, y - 11)
        ctx.lineTo(x + 6, y + 17)
        ctx.stroke()
      }
    })
  }, [phases, layers, time])

  const initThree = useCallback(() => {
    if (!threeRef.current) return

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x0a0a0a)
    sceneRef.current = scene

    const camera = new THREE.PerspectiveCamera(50, 620 / 410, 0.1, 200)
    camera.position.set(4, 26, 72)

    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setSize(620, 410)
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio))
    threeRef.current.innerHTML = ''
    threeRef.current.appendChild(renderer.domElement)
    rendererRef.current = renderer

    const geo = new THREE.PlaneGeometry(128, 64, 96, 96)
    const mat = new THREE.MeshPhongMaterial({
      color: 0x22d3ee,
      emissive: 0x112233,
      shininess: 8,
      specular: 0x445566,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.75
    })

    const mesh = new THREE.Mesh(geo, mat)
    mesh.rotation.x = -Math.PI * 0.42
    scene.add(mesh)
    waveMeshRef.current = mesh

    const light = new THREE.PointLight(0xff33cc, 3, 300)
    light.position.set(40, 90, 60)
    scene.add(light)
    scene.add(new THREE.AmbientLight(0x99aabb, 0.6))

    const loop = () => {
      if (!waveMeshRef.current) return
      const pos = waveMeshRef.current.geometry.attributes.position as THREE.BufferAttribute
      const t = Date.now() * 0.0012

      for (let i = 0; i < pos.count; i++) {
        const x = pos.getX(i)
        const z = pos.getZ(i)
        const y = Math.sin(x * 0.031 + t) * 2.2 + 
                 Math.cos(z * 0.027 + t * 1.3) * 1.8 + 
                 Math.sin((x + z) * 0.019 + t * 0.6) * 1.1
        pos.setY(i, y)
      }
      pos.needsUpdate = true
      renderer.render(scene, camera)
      animationFrameRef.current = requestAnimationFrame(loop)
    }
    loop()
  }, [])

  useEffect(() => {
    let raf: number
    const tick = () => {
      drawCanvas()
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [drawCanvas])

  useEffect(() => {
    if (isRunning) {
      const id = setInterval(() => {
        const newTime = time + 0.072
        setTime(newTime)
        simulateStep(newTime, contextLength)
      }, 36)
      return () => clearInterval(id)
    }
  }, [isRunning, time, contextLength, simulateStep])

  useEffect(() => {
    const t = setTimeout(initThree, 200)
    return () => clearTimeout(t)
  }, [initThree])

  const toggle = () => setIsRunning(v => !v)
  const resetSim = () => {
    setIsRunning(false)
    setTime(0)
    setLayers([])
    setPhases([])
  }

  return (
    <div className="min-h-screen bg-zinc-950 text-white">
      <div className="max-w-screen-2xl mx-auto">
        <header className="flex items-center justify-between px-10 py-8 border-b border-white/10">
          <div className="flex items-center gap-x-4">
            <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-gradient-to-br from-cyan-400 via-violet-500 to-fuchsia-500">
              <Zap className="h-6 w-6" />
            </div>
            <div>
              <div className="text-5xl font-semibold tracking-tighter">WFR</div>
              <div className="text-xs text-zinc-500 -mt-1 font-mono">WAVE • FRACTAL • RESONANT</div>
            </div>
          </div>

          <div className="flex items-center gap-5 text-xs uppercase tracking-widest font-mono text-white/50">
            <div className={`px-4 py-2 rounded-full flex items-center gap-2 ${isRunning ? 'bg-emerald-950 text-emerald-400' : 'bg-zinc-900'}`}>
              <div className={`h-2 w-2 rounded-full ${isRunning ? 'animate-pulse bg-emerald-400' : 'bg-zinc-700'}`} />
              {isRunning ? 'RESONATING' : 'STANDBY'}
            </div>
            <div>PROTOTYPE  •  MARCH 2026</div>
          </div>
        </header>

        <div className="flex">
          {/* Sidebar */}
          <div className="w-80 border-r border-white/10 p-10 space-y-10">
            <div>
              <div className="uppercase text-xs tracking-[1.5px] text-zinc-500 mb-5">ARCHITECTURE CONTROLS</div>
              
              <div className="space-y-8">
                <div>
                  <div className="flex justify-between text-xs mb-3 text-zinc-400">
                    <span>CONTEXT</span>
                    <span className="font-mono">{contextLength}</span>
                  </div>
                  <input type="range" min="256" max="1048576" step="256" 
                         value={contextLength} onChange={e => setContextLength(+e.target.value)}
                         className="w-full accent-cyan-400" />
                </div>

                <div>
                  <div className="flex justify-between text-xs mb-3 text-zinc-400">
                    <span>LEVELS</span>
                    <span className="font-mono">{levels}</span>
                  </div>
                  <input type="range" min="4" max="14" value={levels} 
                         onChange={e => setLevels(+e.target.value)} className="w-full accent-violet-400" />
                </div>

                <div>
                  <div className="flex justify-between text-xs mb-3 text-zinc-400">
                    <span>PHASES</span>
                    <span className="font-mono">{phaseCount}</span>
                  </div>
                  <input type="range" min="6" max="24" value={phaseCount} 
                         onChange={e => setPhaseCount(+e.target.value)} className="w-full accent-fuchsia-400" />
                </div>
              </div>
            </div>

            <div className="pt-8 border-t border-white/10">
              <div className="bg-zinc-900/70 rounded-3xl p-7 text-sm">
                <div className="flex justify-between py-3 border-b border-white/10">
                  <div className="text-zinc-400">CURRENT LOSS</div>
                  <div className="font-mono text-emerald-400">{loss.toFixed(3)}</div>
                </div>
                <div className="flex justify-between py-3">
                  <div className="text-zinc-400">SPIKE ACTIVITY</div>
                  <div className="font-mono text-amber-400">{spikeRate.toFixed(2)}</div>
                </div>
              </div>
            </div>

            <div className="flex gap-4 pt-4">
              <button onClick={toggle} 
                className={`flex-1 py-6 rounded-3xl font-medium flex items-center justify-center gap-3 transition-all ${isRunning ? 'bg-white text-black' : 'bg-white/10 hover:bg-white/20 border border-white/30'}`}>
                {isRunning ? <Pause className="w-5 h-5"/> : <Play className="w-5 h-5"/>}
                {isRunning ? 'PAUSE' : 'START RESONANCE'}
              </button>
              <button onClick={resetSim} className="px-6 rounded-3xl border border-white/20 hover:bg-white/5">
                <RotateCcw className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* Visualizations */}
          <div className="flex-1 p-10">
            <div>
              <div className="flex items-end gap-4 mb-2">
                <h1 className="text-6xl font-bold tracking-tighter">Standing Waves</h1>
                <div className="text-xs bg-white/10 px-5 py-1 rounded-full">LIVE PHASE INTERFERENCE</div>
              </div>
              <p className="text-zinc-400 max-w-md">WFR demonstrates practically infinite context through wave phase encoding, fractal resonance propagation, and event-driven spiking.</p>
            </div>

            <div className="mt-12 grid grid-cols-5 gap-6">
              <div className="col-span-3 bg-zinc-950 border border-white/10 rounded-3xl overflow-hidden">
                <div className="px-8 py-4 bg-black/40 text-xs font-mono border-b border-white/10 flex items-center justify-between">
                  <span>3D RESONANT FIELD</span>
                  <span className="text-cyan-400">PHOTONIC SIMULATION</span>
                </div>
                <div ref={threeRef} className="h-[430px]" />
              </div>

              <div className="col-span-2 bg-zinc-950 border border-white/10 rounded-3xl overflow-hidden flex flex-col">
                <div className="px-8 py-4 bg-black/40 text-xs font-mono border-b border-white/10">2D PHASE + SPIKE VISUALIZATION</div>
                <canvas ref={canvasRef} width="720" height="400" className="flex-1 bg-black" />
              </div>
            </div>

            <div className="mt-10">
              <div className="uppercase text-xs text-white/40 mb-4 tracking-widest">RESONANCE LAYERS • EVENT DRIVEN ACTIVITY</div>
              <div className="grid grid-cols-8 gap-3">
                {layers.map((l, idx) => (
                  <motion.div
                    key={idx}
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    className="bg-zinc-900 border border-white/10 rounded-2xl p-5 text-center"
                  >
                    <div className="text-[10px] text-white/40">LEVEL {l.level}</div>
                    <div className="text-5xl font-light text-white mt-2 tabular-nums tracking-tighter">{l.activeSpikes}</div>
                    <div className="text-xs text-white/50">spikes</div>
                    <div className="mt-6 h-px bg-white/10" />
                    <div className="text-[10px] text-white/30 mt-4">AMP {l.avgAmplitude.toFixed(2)}</div>
                  </motion.div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default WFRAVisualizer
