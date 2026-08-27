import React from "react";
import { Sparkles, CheckCircle2, Clock, Lock } from "lucide-react";

/**
 * MasteryLegend: HUD showing concept status definitions and current session metrics.
 */
export function MasteryLegend({ stats = { mastered: 8, inProgress: 4, locked: 5 } }) {
  return (
    <div className="glass-card px-4 py-2.5 rounded-2xl flex items-center gap-5 border border-slate-800/80 shadow-xl pointer-events-auto">
      {/* Mastered */}
      <div className="flex items-center gap-2">
        <span className="w-2.5 h-2.5 rounded-full bg-amber-400 shadow-[0_0_8px_#f0b429] animate-pulse" />
        <span className="text-xs font-semibold text-amber-300">Mastered</span>
        <span className="text-xs font-mono font-bold text-slate-300 bg-amber-500/15 px-1.5 py-0.5 rounded">
          {stats.mastered}
        </span>
      </div>

      <div className="w-[1px] h-3.5 bg-slate-700/60" />

      {/* In Progress */}
      <div className="flex items-center gap-2">
        <span className="w-2.5 h-2.5 rounded-full bg-blue-500 shadow-[0_0_8px_#3b82f6] animate-pulse" />
        <span className="text-xs font-semibold text-blue-300">In Progress</span>
        <span className="text-xs font-mono font-bold text-slate-300 bg-blue-500/15 px-1.5 py-0.5 rounded">
          {stats.inProgress}
        </span>
      </div>

      <div className="w-[1px] h-3.5 bg-slate-700/60" />

      {/* Locked */}
      <div className="flex items-center gap-2">
        <span className="w-2.5 h-2.5 rounded-full bg-slate-500 border border-slate-400/40" />
        <span className="text-xs font-semibold text-slate-400">Locked</span>
        <span className="text-xs font-mono font-bold text-slate-400 bg-slate-800/40 px-1.5 py-0.5 rounded">
          {stats.locked}
        </span>
      </div>
    </div>
  );
}

export default MasteryLegend;
