import React from "react";
import KnowledgeConstellation from "./components/KnowledgeConstellation";
import MentorOrb from "./components/MentorOrb";
import ChatPanel from "./components/ChatPanel";
import MasteryLegend from "./components/MasteryLegend";
import TopNav from "./components/TopNav";
import useTutorSession from "./hooks/useTutorSession";
import { Sparkles, Brain, Award, AlertTriangle } from "lucide-react";

export function App() {
  const {
    messages,
    pedagogyMode,
    hintLevel,
    hintBudgetRemaining,
    targetConcept,
    isThinking,
    concepts,
    edges,
    constellationRef,
    sendMessage,
    requestHint,
    handleNodeClick,
  } = useTutorSession();

  // Compute live concept stats for the legend
  const stats = React.useMemo(() => {
    return {
      mastered: concepts.filter((c) => c.status === "mastered").length,
      inProgress: concepts.filter((c) => c.status === "in_progress").length,
      locked: concepts.filter((c) => c.status === "locked").length,
    };
  }, [concepts]);

  return (
    <div className="relative w-screen h-screen overflow-hidden bg-[#0a0a0f] text-slate-100 flex flex-col font-sans select-none">
      {/* 1. Full-Bleed 3D Three.js Concept Constellation Background */}
      <KnowledgeConstellation
        ref={constellationRef}
        nodes={concepts}
        edges={edges}
        onNodeClick={handleNodeClick}
      />

      {/* 2. Top Navigation Bar */}
      <TopNav
        pedagogyMode={pedagogyMode}
        hintLevel={hintLevel}
        courseName="Neural Networks & Backpropagation"
      />

      {/* 3. Main Interactive Workspace Split */}
      <main className="relative z-10 flex-1 flex items-stretch justify-between p-6 pt-20 pb-16 gap-6 pointer-events-none">
        {/* Left Side: 3D Mentor Orb & Cognitive State Visualizer */}
        <div className="flex-1 flex flex-col items-center justify-center pointer-events-none">
          <div className="flex flex-col items-center gap-4">
            {/* Centered Reactive 3D Particle Orb */}
            <MentorOrb
              pedagogyMode={pedagogyMode}
              hintLevel={hintLevel}
              size={260}
              className="drop-shadow-2xl pointer-events-auto"
            />

            {/* Orb Mode Badge */}
            <div className="glass-card px-4 py-1.5 rounded-full flex items-center gap-2 pointer-events-auto border border-slate-700/60 shadow-xl">
              <span
                className={`w-2 h-2 rounded-full ${
                  pedagogyMode === "hint"
                    ? "bg-amber-400 animate-pulse shadow-[0_0_10px_#f0b429]"
                    : pedagogyMode === "thinking"
                    ? "bg-blue-400 animate-ping shadow-[0_0_10px_#3b82f6]"
                    : pedagogyMode === "celebrate"
                    ? "bg-emerald-400 animate-bounce shadow-[0_0_10px_#10b981]"
                    : pedagogyMode === "stuck"
                    ? "bg-indigo-400 shadow-[0_0_10px_#818cf8]"
                    : "bg-blue-400 shadow-[0_0_6px_#3b82f6]"
                }`}
              />
              <span className="text-xs font-mono font-semibold uppercase text-slate-300">
                Mentor State: <span className="text-white font-bold">{pedagogyMode}</span>
              </span>
            </div>

            {/* Quick Demo Action Chips */}
            <div className="flex items-center gap-2 pointer-events-auto mt-2">
              <button
                onClick={() => sendMessage("The derivative of the loss with respect to weight is chain rule multiplication.")}
                className="glass-card hover:bg-slate-800/80 text-[11px] text-slate-300 hover:text-white px-3 py-1.5 rounded-xl border border-slate-700/60 transition active:scale-95 flex items-center gap-1.5"
              >
                <Brain className="w-3.5 h-3.5 text-blue-400" />
                Submit Derivation
              </button>
              <button
                onClick={requestHint}
                className="glass-card hover:bg-amber-500/20 text-[11px] text-amber-300 px-3 py-1.5 rounded-xl border border-amber-500/30 transition active:scale-95 flex items-center gap-1.5"
              >
                <Sparkles className="w-3.5 h-3.5 text-amber-400" />
                Trigger Hint
              </button>
              <button
                onClick={() => constellationRef.current?.updateNode("backpropagation", "mastered", 1.0)}
                className="glass-card hover:bg-emerald-500/20 text-[11px] text-emerald-300 px-3 py-1.5 rounded-xl border border-emerald-500/30 transition active:scale-95 flex items-center gap-1.5"
              >
                <Award className="w-3.5 h-3.5 text-emerald-400" />
                Master Backprop
              </button>
            </div>
          </div>
        </div>

        {/* Right Side: Chat Conversation Surface (Glassmorphic HUD) */}
        <div className="w-[460px] max-w-[45vw] h-full pointer-events-auto flex flex-col">
          <ChatPanel
            messages={messages}
            onSendMessage={sendMessage}
            onRequestHint={requestHint}
            hintBudgetRemaining={hintBudgetRemaining}
            currentHintLevel={hintLevel}
            targetConcept={targetConcept}
            isThinking={isThinking}
          />
        </div>
      </main>

      {/* 4. Bottom HUD: Mastery Legend & Constellation Controls */}
      <footer className="fixed bottom-4 left-6 z-20 pointer-events-none">
        <MasteryLegend stats={stats} />
      </footer>
    </div>
  );
}

export default App;
