import React from "react";
import { Atom, Compass, Activity, Sparkles, Terminal } from "lucide-react";

export function TopNav({
  pedagogyMode = "idle",
  hintLevel = 0,
  courseName = "Machine Learning & Deep Neural Systems",
}) {
  return (
    <header className="fixed top-0 left-0 right-0 z-30 px-6 py-3.5 flex items-center justify-between pointer-events-none">
      {/* Brand & Course Title */}
      <div className="glass-card px-4 py-2 rounded-2xl flex items-center gap-3 border border-slate-800/80 shadow-2xl pointer-events-auto">
        <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-amber-500 to-blue-600 flex items-center justify-center shadow-[0_0_15px_rgba(59,130,246,0.5)]">
          <Atom className="w-5 h-5 text-white animate-spin-slow" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-sm font-extrabold tracking-tight text-slate-100">
              AETHER <span className="text-amber-400 font-normal">TUTOR</span>
            </h1>
            <span className="text-[10px] font-mono px-1.5 py-0.2 bg-slate-800 text-slate-300 rounded border border-slate-700">
              v1.0
            </span>
          </div>
          <p className="text-[11px] text-slate-400 font-medium">{courseName}</p>
        </div>
      </div>

      {/* Realtime Pedagogical Status Monitor */}
      <div className="glass-card px-4 py-2 rounded-2xl flex items-center gap-4 border border-slate-800/80 shadow-2xl pointer-events-auto">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-blue-400" />
          <span className="text-xs text-slate-400 font-medium">Cognitive Mode:</span>
          <span
            className={`text-xs font-mono font-bold uppercase px-2 py-0.5 rounded-md ${
              pedagogyMode === "hint"
                ? "bg-amber-500/20 text-amber-300 border border-amber-500/30"
                : pedagogyMode === "thinking"
                ? "bg-blue-500/20 text-blue-300 border border-blue-500/30"
                : pedagogyMode === "celebrate"
                ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                : pedagogyMode === "stuck"
                ? "bg-indigo-500/20 text-indigo-300 border border-indigo-500/30"
                : "bg-slate-800 text-slate-300 border border-slate-700"
            }`}
          >
            {pedagogyMode}
          </span>
        </div>

        {hintLevel > 0 && (
          <div className="flex items-center gap-1 text-xs text-amber-300 font-mono">
            <span>Hint Scaffolding:</span>
            <span className="font-bold bg-amber-500/20 px-1.5 py-0.5 rounded">L{hintLevel}</span>
          </div>
        )}
      </div>
    </header>
  );
}

export default TopNav;
