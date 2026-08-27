import React from "react";
import { Lightbulb, Lock, HelpCircle } from "lucide-react";

/**
 * HintRevealButton:
 * Lets the student request the next hint level; disabled once hint_budget_remaining
 * hits 0, explicitly displaying the pedagogical reason why.
 */
export function HintRevealButton({
  hintBudgetRemaining = 3,
  currentHintLevel = 0,
  onRequestHint,
  isLoading = false,
}) {
  const isBudgetExhausted = hintBudgetRemaining <= 0;

  return (
    <div className="relative group inline-flex items-center">
      <button
        onClick={onRequestHint}
        disabled={isBudgetExhausted || isLoading}
        className={`flex items-center gap-2.5 px-4 py-2 rounded-xl text-xs font-semibold tracking-wide transition-all duration-300 ${
          isBudgetExhausted
            ? "bg-slate-800/60 text-slate-500 border border-slate-700/50 cursor-not-allowed"
            : "bg-gradient-to-r from-amber-500/20 to-amber-600/20 hover:from-amber-500/30 hover:to-amber-600/30 text-amber-300 border border-amber-500/40 hover:border-amber-400/70 hover:shadow-[0_0_20px_-3px_rgba(240,180,41,0.4)] active:scale-95"
        }`}
      >
        {isBudgetExhausted ? (
          <Lock className="w-4 h-4 text-slate-500" />
        ) : (
          <Lightbulb
            className={`w-4 h-4 ${
              currentHintLevel > 0 ? "text-amber-400 fill-amber-400/30 animate-pulse" : "text-amber-300"
            }`}
          />
        )}

        <span>
          {isBudgetExhausted
            ? "Hint Budget Exhausted"
            : currentHintLevel > 0
            ? `Request Hint Level ${currentHintLevel + 1}`
            : "Need a Hint?"}
        </span>

        {/* Budget Counter Badge */}
        <span
          className={`ml-1 px-1.5 py-0.5 rounded-md text-[10px] font-mono font-bold ${
            isBudgetExhausted
              ? "bg-slate-700/50 text-slate-400"
              : "bg-amber-400/20 text-amber-300 border border-amber-400/30"
          }`}
        >
          {hintBudgetRemaining} left
        </span>
      </button>

      {/* Explanatory Tooltip when Budget Exhausted */}
      {isBudgetExhausted && (
        <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 hidden group-hover:flex flex-col items-center z-50 pointer-events-none">
          <div className="glass-card px-3 py-2 rounded-lg text-[11px] text-slate-300 text-center max-w-xs shadow-2xl border border-slate-700/70">
            <div className="flex items-center gap-1.5 font-semibold text-amber-400 mb-0.5 justify-center">
              <HelpCircle className="w-3.5 h-3.5" />
              <span>Diagnostic Limit Reached</span>
            </div>
            <p className="text-slate-400">
              You've utilized all hints for this step. Try attempting an answer or ask to explain the prerequisite!
            </p>
          </div>
          <div className="w-2 h-2 rotate-45 bg-[#161622] border-r border-b border-slate-700/70 -mt-1" />
        </div>
      )}
    </div>
  );
}

export default HintRevealButton;
