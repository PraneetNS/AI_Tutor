import React, { useRef, useMemo, useEffect } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { animate } from "animejs";

/**
 * 3D Particle System for MentorOrb.
 * Smoothly interpolated via anime.js parameters when pedagogyMode or hintLevel changes.
 */
function ParticleOrbMesh({ pedagogyMode = "idle", hintLevel = 0 }) {
  const pointsRef = useRef();
  const particleCount = 1200;

  // Mutable animated parameters driven by anime.js
  const animState = useRef({
    rotationSpeed: 0.4,
    particleSpread: 1.0,
    r: 59 / 255,  // #3b82f6 (blue)
    g: 130 / 255,
    b: 246 / 255,
    size: 0.065,
    pulseSpeed: 1.0,
    noiseAmp: 0.15,
  });

  // Mode Target Configurations
  const modeConfigs = useMemo(() => ({
    idle: {
      rotationSpeed: 0.4,
      particleSpread: 1.0,
      color: [59 / 255, 130 / 255, 246 / 255], // Soft blue
      size: 0.065,
      pulseSpeed: 1.0,
      noiseAmp: 0.12,
    },
    thinking: {
      rotationSpeed: 2.2, // Rapid orbital vortex
      particleSpread: 1.25,
      color: [96 / 255, 165 / 255, 250 / 255], // Bright cyan-blue
      size: 0.08,
      pulseSpeed: 3.5,
      noiseAmp: 0.35,
    },
    hint: {
      rotationSpeed: 0.8,
      // Higher hint levels compress particle spread
      particleSpread: Math.max(0.65, 1.0 - hintLevel * 0.08),
      color: [240 / 255, 180 / 255, 41 / 255], // Warm gold (#f0b429)
      size: 0.075 + hintLevel * 0.01,
      pulseSpeed: 1.8 + hintLevel * 0.3,
      noiseAmp: 0.2,
    },
    celebrate: {
      rotationSpeed: 1.6,
      particleSpread: 1.65, // Expansive burst
      color: [251 / 255, 191 / 255, 36 / 255], // Golden corona
      size: 0.09,
      pulseSpeed: 4.0,
      noiseAmp: 0.4,
    },
    stuck: {
      rotationSpeed: 0.2, // Slow dense contraction
      particleSpread: 0.75,
      color: [99 / 255, 102 / 255, 241 / 255], // Deep indigo
      size: 0.055,
      pulseSpeed: 0.6,
      noiseAmp: 0.08,
    },
  }), [hintLevel]);

  // Smooth anime.js transition whenever pedagogyMode or hintLevel updates (~600ms)
  useEffect(() => {
    const target = modeConfigs[pedagogyMode] || modeConfigs.idle;

    animate(animState.current, {
      rotationSpeed: target.rotationSpeed,
      particleSpread: target.particleSpread,
      r: target.color[0],
      g: target.color[1],
      b: target.color[2],
      size: target.size,
      pulseSpeed: target.pulseSpeed,
      noiseAmp: target.noiseAmp,
      duration: 600,
      ease: "inOutCubic",
    });
  }, [pedagogyMode, hintLevel, modeConfigs]);

  // Generate initial Fibonacci sphere particle distributions
  const [basePositions, randomPhases] = useMemo(() => {
    const positions = new Float32Array(particleCount * 3);
    const phases = new Float32Array(particleCount);

    const phi = Math.PI * (3 - Math.sqrt(5)); // Golden ratio angle

    for (let i = 0; i < particleCount; i++) {
      const y = 1 - (i / (particleCount - 1)) * 2; // y from 1 to -1
      const radius = Math.sqrt(1 - y * y);
      const theta = phi * i;

      const x = Math.cos(theta) * radius;
      const z = Math.sin(theta) * radius;

      positions[i * 3] = x * 1.8;
      positions[i * 3 + 1] = y * 1.8;
      positions[i * 3 + 2] = z * 1.8;

      phases[i] = Math.random() * Math.PI * 2;
    }

    return [positions, phases];
  }, [particleCount]);

  // Animated vertex positions buffer
  const currentPositions = useMemo(() => new Float32Array(basePositions), [basePositions]);

  // Render loop with particle harmonics
  useFrame(({ clock }) => {
    if (!pointsRef.current) return;
    const t = clock.getElapsedTime();
    const state = animState.current;

    // Rotate the particle sphere
    pointsRef.current.rotation.y = t * state.rotationSpeed;
    pointsRef.current.rotation.x = Math.sin(t * 0.5) * 0.2;

    // Update particle positions with harmonic radial pulsing
    const posAttr = pointsRef.current.geometry.attributes.position;
    const arr = posAttr.array;

    for (let i = 0; i < particleCount; i++) {
      const idx = i * 3;
      const bx = basePositions[idx];
      const by = basePositions[idx + 1];
      const bz = basePositions[idx + 2];

      const phase = randomPhases[i];
      const pulse = 1.0 + Math.sin(t * state.pulseSpeed + phase) * state.noiseAmp;
      const scale = state.particleSpread * pulse;

      arr[idx] = bx * scale;
      arr[idx + 1] = by * scale;
      arr[idx + 2] = bz * scale;
    }

    posAttr.needsUpdate = true;

    // Update particle material color & size dynamically
    if (pointsRef.current.material) {
      pointsRef.current.material.color.setRGB(state.r, state.g, state.b);
      pointsRef.current.material.size = state.size;
    }
  });

  return (
    <group>
      {/* Central Inner Glow Core */}
      <mesh>
        <sphereGeometry args={[0.9, 24, 24]} />
        <meshBasicMaterial
          color={new THREE.Color(animState.current.r, animState.current.g, animState.current.b)}
          transparent
          opacity={0.12}
          blending={THREE.AdditiveBlending}
        />
      </mesh>

      {/* Main Reactive Particle Points */}
      <points ref={pointsRef}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={particleCount}
            array={currentPositions}
            itemSize={3}
          />
        </bufferGeometry>
        <pointsMaterial
          size={0.065}
          color={new THREE.Color("#3b82f6")}
          transparent
          opacity={0.85}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
        />
      </points>
    </group>
  );
}

/**
 * MentorOrb: Centered interactive 3D particle sphere.
 * Decoupled component ready for future audio / voice reactivity without touching ChatPanel.
 */
export function MentorOrb({
  pedagogyMode = "idle",
  hintLevel = 0,
  className = "",
  size = 180,
}) {
  return (
    <div
      className={`relative flex items-center justify-center pointer-events-none select-none ${className}`}
      style={{ width: size, height: size }}
    >
      {/* Background radial gradient bloom */}
      <div
        className={`absolute inset-0 rounded-full blur-2xl transition-all duration-700 pointer-events-none ${
          pedagogyMode === "hint" || pedagogyMode === "celebrate"
            ? "bg-amber-500/25"
            : pedagogyMode === "thinking"
            ? "bg-blue-500/30"
            : pedagogyMode === "stuck"
            ? "bg-indigo-600/25"
            : "bg-blue-500/15"
        }`}
      />

      <Canvas
        camera={{ position: [0, 0, 4.5], fov: 45 }}
        gl={{ antialias: true, alpha: true }}
      >
        <ambientLight intensity={0.5} />
        <ParticleOrbMesh pedagogyMode={pedagogyMode} hintLevel={hintLevel} />
      </Canvas>
    </div>
  );
}

export default MentorOrb;
