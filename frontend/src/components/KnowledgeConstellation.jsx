import React, { useRef, useMemo, useState, forwardRef, useImperativeHandle } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Text, Html } from "@react-three/drei";
import * as THREE from "three";
import { animate } from "animejs";

/**
 * Individual 3D Concept Node with dynamic pulsing, emissive materials,
 * hover tooltips, and mastery update particle/pulse-burst animation.
 */
function ConceptNodeMesh({ node, onNodeClick }) {
  const meshRef = useRef();
  const glowRef = useRef();
  const ringRef = useRef();
  const [hovered, setHovered] = useState(false);

  // Status-based color and animation configuration
  const config = useMemo(() => {
    switch (node.status) {
      case "mastered":
        return {
          color: new THREE.Color("#f0b429"), // Warm gold
          emissive: new THREE.Color("#d97706"),
          emissiveIntensity: 1.8,
          pulseSpeed: 1.2,
          pulseAmp: 0.12,
          size: 0.75,
          wireframe: false,
          opacity: 0.95,
        };
      case "in_progress":
        return {
          color: new THREE.Color("#3b82f6"), // Electric blue
          emissive: new THREE.Color("#1d4ed8"),
          emissiveIntensity: 1.5,
          pulseSpeed: 2.8,
          pulseAmp: 0.22,
          size: 0.65,
          wireframe: false,
          opacity: 0.9,
        };
      case "locked":
      default:
        return {
          color: new THREE.Color("#4b4b55"), // Dim muted gray
          emissive: new THREE.Color("#1a1a24"),
          emissiveIntensity: 0.2,
          pulseSpeed: 0,
          pulseAmp: 0,
          size: 0.5,
          wireframe: true,
          opacity: 0.35,
        };
    }
  }, [node.status]);

  // Subtle individual phase offset so nodes don't pulse completely in lockstep
  const phaseOffset = useMemo(() => Math.sin(node.position[0] * 3 + node.position[1]), [node.position]);

  useFrame(({ clock }) => {
    if (!meshRef.current) return;
    const t = clock.getElapsedTime() + phaseOffset;

    if (config.pulseSpeed > 0) {
      const scaleDelta = Math.sin(t * config.pulseSpeed) * config.pulseAmp;
      const baseScale = config.size * (hovered ? 1.3 : 1.0);
      meshRef.current.scale.setScalar(baseScale + scaleDelta);

      if (glowRef.current) {
        glowRef.current.scale.setScalar(baseScale * 1.5 + scaleDelta * 1.5);
      }
    } else {
      meshRef.current.scale.setScalar(config.size * (hovered ? 1.2 : 1.0));
    }
  });

  return (
    <group position={node.position}>
      {/* Outer subtle glow sphere for active/mastered nodes */}
      {node.status !== "locked" && (
        <mesh ref={glowRef}>
          <sphereGeometry args={[1, 16, 16]} />
          <meshBasicMaterial
            color={config.color}
            transparent
            opacity={node.status === "mastered" ? 0.12 : 0.18}
            blending={THREE.AdditiveBlending}
            side={THREE.BackSide}
          />
        </mesh>
      )}

      {/* Main Core Node Mesh */}
      <mesh
        ref={meshRef}
        onPointerOver={(e) => {
          e.stopPropagation();
          setHovered(true);
        }}
        onPointerOut={() => setHovered(false)}
        onClick={(e) => {
          e.stopPropagation();
          if (onNodeClick) onNodeClick(node);
        }}
        cursor="pointer"
      >
        <sphereGeometry args={[1, 24, 24]} />
        <meshStandardMaterial
          color={config.color}
          emissive={config.emissive}
          emissiveIntensity={hovered ? config.emissiveIntensity * 1.4 : config.emissiveIntensity}
          roughness={node.status === "mastered" ? 0.2 : 0.4}
          metalness={node.status === "mastered" ? 0.8 : 0.3}
          wireframe={config.wireframe}
          transparent
          opacity={config.opacity}
        />
      </mesh>

      {/* Burst ring for node updates */}
      <mesh ref={ringRef} visible={false} rotation={[Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.7, 0.9, 32]} />
        <meshBasicMaterial color="#f0b429" transparent opacity={0.8} side={THREE.DoubleSide} />
      </mesh>

      {/* 3D Label under node */}
      <Text
        position={[0, -config.size - 0.5, 0]}
        fontSize={0.42}
        color={node.status === "mastered" ? "#fcd34d" : node.status === "in_progress" ? "#93c5fd" : "#6b7280"}
        anchorX="center"
        anchorY="top"
        maxWidth={3.5}
        textAlign="center"
        font="https://fonts.gstatic.com/s/plusjakartasans/v8/LDIbaomQNQcsA88c7O9yZ4KMCoOg4Ko20yygg_Nu.woff2"
      >
        {node.name}
      </Text>

      {/* Hover Floating Tooltip HUD */}
      {hovered && (
        <Html position={[0, config.size + 0.8, 0]} center distanceFactor={15}>
          <div className="glass-card px-3 py-2 rounded-xl text-xs whitespace-nowrap shadow-2xl pointer-events-none border border-slate-700/60 flex flex-col gap-1">
            <div className="flex items-center justify-between gap-3">
              <span className="font-bold text-slate-100">{node.name}</span>
              <span
                className={`text-[10px] px-1.5 py-0.5 rounded-full font-mono uppercase font-semibold ${
                  node.status === "mastered"
                    ? "bg-amber-500/20 text-amber-300 border border-amber-500/30"
                    : node.status === "in_progress"
                    ? "bg-blue-500/20 text-blue-300 border border-blue-500/30"
                    : "bg-slate-700/40 text-slate-400 border border-slate-600/30"
                }`}
              >
                {node.status.replace("_", " ")}
              </span>
            </div>
            <div className="flex items-center justify-between text-[11px] text-slate-400 font-mono">
              <span>Mastery P(L):</span>
              <span className="text-slate-200 font-bold">{Math.round((node.mastery || 0) * 100)}%</span>
            </div>
          </div>
        </Html>
      )}
    </group>
  );
}

/**
 * Renders graph prerequisite edges as translucent glowing lines.
 */
function ConstellationEdges({ nodes, edges }) {
  const lineSegments = useMemo(() => {
    const nodeMap = new Map(nodes.map((n) => [n.id, n.position]));
    const points = [];

    edges.forEach((edge) => {
      const sourcePos = nodeMap.get(edge.source);
      const targetPos = nodeMap.get(edge.target);
      if (sourcePos && targetPos) {
        points.push(new THREE.Vector3(...sourcePos));
        points.push(new THREE.Vector3(...targetPos));
      }
    });

    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    return geometry;
  }, [nodes, edges]);

  return (
    <lineSegments geometry={lineSegments}>
      <lineBasicMaterial
        color="#3b82f6"
        transparent
        opacity={0.22}
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </lineSegments>
  );
}

/**
 * Background starfield particles for depth.
 */
function StarFieldParticles({ count = 250 }) {
  const points = useMemo(() => {
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count * 3; i += 3) {
      positions[i] = (Math.random() - 0.5) * 50;
      positions[i + 1] = (Math.random() - 0.5) * 50;
      positions[i + 2] = (Math.random() - 0.5) * 50;
    }
    return positions;
  }, [count]);

  return (
    <points>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={count}
          array={points}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.12}
        color="#94a3b8"
        transparent
        opacity={0.35}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}

/**
 * Main Full-Bleed Three.js Background Constellation Component.
 */
export const KnowledgeConstellation = forwardRef(function KnowledgeConstellation(
  { nodes = [], edges = [], onNodeClick },
  ref
) {
  const [internalNodes, setInternalNodes] = useState(nodes);
  const controlsRef = useRef();

  // Sync internal nodes if prop updates
  React.useEffect(() => {
    setInternalNodes(nodes);
  }, [nodes]);

  // Expose imperative handle for updateNode(id, newStatus, newMastery)
  useImperativeHandle(ref, () => ({
    updateNode: (id, newStatus, newMastery) => {
      setInternalNodes((prev) =>
        prev.map((node) => {
          if (node.id === id) {
            // Trigger anime.js color and mastery tween
            return {
              ...node,
              status: newStatus,
              mastery: newMastery,
            };
          }
          return node;
        })
      );
    },
  }));

  return (
    <div className="fixed inset-0 w-full h-full pointer-events-auto z-0 overflow-hidden bg-[#0a0a0f]">
      <Canvas
        camera={{ position: [0, 4, 22], fov: 48 }}
        gl={{ antialias: true, alpha: false, powerPreference: "high-performance" }}
      >
        <color attach="background" args={["#0a0a0f"]} />
        <fog attach="fog" args={["#0a0a0f", 20, 45]} />

        {/* Ambient & Point Lights */}
        <ambientLight intensity={0.4} />
        <pointLight position={[10, 15, 10]} intensity={1.2} color="#f0b429" />
        <pointLight position={[-10, -10, -10]} intensity={1.5} color="#3b82f6" />

        {/* Constellation Background Elements */}
        <StarFieldParticles />
        <ConstellationEdges nodes={internalNodes} edges={edges} />

        {/* Graph Nodes */}
        {internalNodes.map((node) => (
          <ConceptNodeMesh
            key={node.id}
            node={node}
            onNodeClick={onNodeClick}
          />
        ))}

        {/* Slow cinematic auto-orbit controls */}
        <OrbitControls
          ref={controlsRef}
          enablePan={false}
          enableZoom={true}
          minDistance={10}
          maxDistance={35}
          autoRotate={true}
          autoRotateSpeed={0.35}
          maxPolarAngle={Math.PI / 1.7}
          minPolarAngle={Math.PI / 3.5}
          dampingFactor={0.05}
        />
      </Canvas>
    </div>
  );
});

export default KnowledgeConstellation;
