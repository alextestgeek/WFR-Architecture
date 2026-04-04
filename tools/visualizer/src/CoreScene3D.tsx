import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

export interface LayerRow {
  level: number
  spikeRate: number
  silentPct: number
  avgAmplitude: number
  resonances: number[]
  spikes: boolean[]
}

export interface WFRResultLive {
  phases: number[][]
  layers: LayerRow[]
  rc: number
  standingWave: number[]
  contextLength: number
}

type ThemeT = { bg: string; border: string; accent: string }

const LAYER_COLORS = [0x818cf8, 0x34d399, 0xfbbf24, 0xf87171, 0xa78bfa, 0x2dd4bf, 0xfb923c, 0xe879f9]

export function CoreScene3D({ result, theme }: { result: WFRResultLive; theme: ThemeT }) {
  const mountRef = useRef<HTMLDivElement>(null)
  const resultRef = useRef(result)
  resultRef.current = result

  useEffect(() => {
    const mount = mountRef.current
    if (!mount) return

    const width = mount.clientWidth
    const height = mount.clientHeight
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(theme.bg)

    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 200)
    camera.position.set(6, 4.5, 9)

    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setSize(width, height)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    mount.appendChild(renderer.domElement)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.target.set(0, 1.2, 0)

    scene.add(new THREE.AmbientLight(0xffffff, 0.35))
    const dir = new THREE.DirectionalLight(0xffffff, 0.9)
    dir.position.set(4, 10, 6)
    scene.add(dir)

    const wpeMat = new THREE.MeshStandardMaterial({
      color: 0x22d3ee,
      emissive: 0x0e7490,
      emissiveIntensity: 0.45,
      metalness: 0.2,
      roughness: 0.45,
    })
    const wpe = new THREE.Mesh(new THREE.SphereGeometry(0.55, 32, 32), wpeMat)
    wpe.position.set(-3.2, 1.2, 0)
    scene.add(wpe)

    const rcMat = new THREE.MeshStandardMaterial({
      color: 0xf472b6,
      emissive: 0x831843,
      emissiveIntensity: 0.35,
      metalness: 0.3,
      roughness: 0.35,
    })
    const rcMesh = new THREE.Mesh(new THREE.IcosahedronGeometry(0.5, 1), rcMat)
    rcMesh.position.set(3.4, 1.2, 0)
    scene.add(rcMesh)

    const maxRings = 16
    const torusGeo = new THREE.TorusGeometry(1.05, 0.07, 12, 48)
    const rings: THREE.Mesh[] = []
    for (let i = 0; i < maxRings; i++) {
      const mat = new THREE.MeshStandardMaterial({
        color: LAYER_COLORS[i % LAYER_COLORS.length],
        emissive: LAYER_COLORS[i % LAYER_COLORS.length],
        emissiveIntensity: 0.25,
        metalness: 0.25,
        roughness: 0.4,
      })
      const mesh = new THREE.Mesh(torusGeo, mat)
      mesh.rotation.x = Math.PI / 2
      mesh.position.set(0, 0.35 + i * 0.52, 0)
      mesh.visible = false
      scene.add(mesh)
      rings.push(mesh)
    }

    const geoWpe = new THREE.BufferGeometry()
    const geoRc = new THREE.BufferGeometry()
    const matWpe = new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.6 })
    const matRc = new THREE.LineBasicMaterial({ color: 0xf472b6, transparent: true, opacity: 0.55 })
    const conWpe = new THREE.Line(geoWpe, matWpe)
    const conRc = new THREE.Line(geoRc, matRc)
    scene.add(conWpe)
    scene.add(conRc)

    let animation = 0
    let raf = 0
    const tick = () => {
      animation += 0.018
      const r = resultRef.current
      let maxAmp = 1e-6
      for (const L of r.layers) if (L.avgAmplitude > maxAmp) maxAmp = L.avgAmplitude

      wpeMat.emissiveIntensity = 0.35 + 0.15 * Math.sin(animation * 2)
      rcMat.emissiveIntensity = 0.2 + 0.55 * Math.min(1, r.rc * 1.3) * (0.5 + 0.5 * Math.sin(animation))

      for (let i = 0; i < rings.length; i++) {
        const mesh = rings[i]
        if (i < r.layers.length) {
          mesh.visible = true
          const ampN = r.layers[i].avgAmplitude / maxAmp
          const mmat = mesh.material as THREE.MeshStandardMaterial
          mmat.emissiveIntensity = 0.12 + 0.88 * ampN
          const s = 0.85 + 0.35 * ampN + 0.2 * r.layers[i].spikeRate
          mesh.scale.set(s, s, s)
        } else mesh.visible = false
      }

      const nL = r.layers.length
      const yLast = nL > 0 ? 0.35 + (nL - 1) * 0.52 + 0.26 : 1.2
      geoWpe.setFromPoints([
        new THREE.Vector3(-3.2, 1.2, 0),
        new THREE.Vector3(0, 0.35 + 0.26, 0),
      ])
      geoRc.setFromPoints([new THREE.Vector3(0, yLast, 0), new THREE.Vector3(3.4, 1.2, 0)])

      controls.update()
      renderer.render(scene, camera)
      raf = requestAnimationFrame(tick)
    }
    tick()

    const onResize = () => {
      if (!mountRef.current) return
      const w = mountRef.current.clientWidth
      const h = mountRef.current.clientHeight
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h)
    }
    window.addEventListener('resize', onResize)

    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', onResize)
      scene.remove(conWpe, conRc)
      geoWpe.dispose()
      geoRc.dispose()
      matWpe.dispose()
      matRc.dispose()
      controls.dispose()
      renderer.dispose()
      mount.removeChild(renderer.domElement)
      torusGeo.dispose()
    }
  }, [theme.bg])

  return (
    <div
      ref={mountRef}
      className="w-full rounded-xl overflow-hidden"
      style={{ height: 320, border: '1px solid ' + theme.border }}
    />
  )
}
