import React, { useRef, useEffect, useState } from "react";
import { Send, Sparkles, BookOpen, Brain, Lightbulb, CheckCircle2, AlertCircle } from "lucide-react";
import { animate } from "animejs";
import HintRevealButton from "./HintRevealButton";

/**
 * Individual Chat Message Card.
 * Renders dynamically according to role, pedagogy_mode, and hint_level.
 */
function ChatMessageItem({ message, index }) {
  const itemRef = useRef();
  const textRef = useRef();

  // Entrance animation (fade + slight vertical rise staggered)
  useEffect(() => {
    if (itemRef.current) {
      animate(itemRef.current, {
        opacity: [0, 1],
        translateY: [16, 0],
        duration: 450,
        delay: index * 60,
        ease: "outQuad",
      });
    }
  }, [index]);

  // Progressive blur-to-focus reveal animation for hints using anime.js
  useEffect(() => {
    if (message.isHint && textRef.current) {
      // Calculate initial blur based on hint level (higher hint level starts clearer)
      const initialBlur = Math.max(0, 10 - (message.hintLevel || 1) * 2.5);
      animate(textRef.current, {
        filter: [`blur(${initialBlur}px)`, "blur(0px)"],
        opacity: [0.4, 1],
        duration: 800,
        ease: "outCubic",
      });
    }
  }, [message.isHint, message.hintLevel]);

  const isUser = message.role === "user" || message.role === "student";
  const mode = message.pedagogyMode || "socratic";

  // 1. User Message Style
  if (isUser) {
    return (
      <div ref={itemRef} className="flex justify-end mb-4">
        <div className="max-w-[85%] bg-blue-600/25 border border-blue-500/30 text-slate-100 rounded-2xl rounded-tr-sm px-4 py-3 shadow-lg">
          <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</p>
        </div>
      </div>
    );
  }

  // 2. Tutor Socratic Question Style (Dashed accent border)
  if (mode === "socratic" || mode === "SOCRATIC" || mode === "guide") {
    return (
      <div ref={itemRef} className="flex justify-start mb-4">
        <div className="max-w-[90%] glass-card rounded-2xl rounded-tl-sm p-4 border-2 border-dashed border-blue-400/60 shadow-[0_0_20px_-5px_rgba(59,130,246,0.2)]">
          <div className="flex items-center gap-2 mb-2">
            <span className="flex items-center gap-1 text-[11px] font-bold font-mono tracking-wider uppercase bg-blue-500/20 text-blue-300 px-2 py-0.5 rounded-md border border-blue-500/30">
              <Brain className="w-3.5 h-3.5 text-blue-400" />
              Socratic Dialogue
            </span>
            {message.targetConcept && (
              <span className="text-[11px] text-slate-400 font-mono">
                Topic: <span className="text-slate-200">{message.targetConcept}</span>
              </span>
            )}
          </div>
          <div ref={textRef}>
            <p className="text-sm leading-relaxed text-slate-200 whitespace-pre-wrap">{message.content}</p>
          </div>
        </div>
      </div>
    );
  }

  // 3. Tutor Hint Style (Progressive blur-to-focus reveal + warm gold theme)
  if (mode === "hint" || mode === "HINT" || message.isHint) {
    return (
      <div ref={itemRef} className="flex justify-start mb-4">
        <div className="max-w-[90%] glass-card rounded-2xl rounded-tl-sm p-4 border border-amber-500/50 shadow-[0_0_25px_-5px_rgba(240,180,41,0.25)] bg-gradient-to-br from-amber-950/30 to-slate-900/60">
          <div className="flex items-center justify-between gap-2 mb-2">
            <span className="flex items-center gap-1.5 text-[11px] font-bold font-mono tracking-wider uppercase bg-amber-500/25 text-amber-300 px-2.5 py-0.5 rounded-md border border-amber-500/40 animate-pulse">
              <Lightbulb className="w-3.5 h-3.5 text-amber-300 fill-amber-300/30" />
              Scaffolding Hint • Level {message.hintLevel || 1}
            </span>
          </div>
          <div ref={textRef} className="relative">
            <p className="text-sm leading-relaxed text-amber-100 font-medium whitespace-pre-wrap">{message.content}</p>
          </div>
        </div>
      </div>
    );
  }

  // 4. Tutor Direct Explanation Style (Solid structured glass card)
  return (
    <div ref={itemRef} className="flex justify-start mb-4">
      <div className="max-w-[90%] glass-card rounded-2xl rounded-tl-sm p-4 border border-slate-700/70 shadow-xl bg-slate-900/80">
        <div className="flex items-center gap-2 mb-2">
          <span className="flex items-center gap-1 text-[11px] font-bold font-mono tracking-wider uppercase bg-emerald-500/20 text-emerald-300 px-2 py-0.5 rounded-md border border-emerald-500/30">
            <BookOpen className="w-3.5 h-3.5 text-emerald-400" />
            Concept Explanation
          </span>
        </div>
        <div ref={textRef}>
          <p className="text-sm leading-relaxed text-slate-200 whitespace-pre-wrap">{message.content}</p>
        </div>
      </div>
    </div>
  );
}

/**
 * Main ChatPanel Component.
 */
export function ChatPanel({
  messages = [],
  onSendMessage,
  onRequestHint,
  hintBudgetRemaining = 3,
  currentHintLevel = 0,
  targetConcept = "Gradient Descent",
  isThinking = false,
}) {
  const [inputVal, setInputVal] = useState("");
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isThinking]);

  const handleSend = (e) => {
    e.preventDefault();
    if (!inputVal.trim() || isThinking) return;
    onSendMessage(inputVal.trim());
    setInputVal("");
  };

  return (
    <div className="glass-panel w-full h-full flex flex-col rounded-2xl overflow-hidden shadow-2xl border border-slate-800/80">
      {/* Panel Header */}
      <div className="px-5 py-3.5 border-b border-slate-800/80 flex items-center justify-between bg-slate-950/40">
        <div className="flex items-center gap-3">
          <div className="relative flex items-center justify-center w-8 h-8 rounded-lg bg-blue-500/10 border border-blue-500/30">
            <Sparkles className="w-4 h-4 text-blue-400" />
            {isThinking && (
              <span className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 bg-amber-400 rounded-full animate-ping" />
            )}
          </div>
          <div>
            <h2 className="text-sm font-bold text-slate-100 flex items-center gap-2">
              Aether Tutor
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-blue-500/15 text-blue-300 font-normal">
                Active Session
              </span>
            </h2>
            <p className="text-[11px] text-slate-400 font-mono">
              Target: <span className="text-amber-400 font-semibold">{targetConcept}</span>
            </p>
          </div>
        </div>

        {/* Hint Action Button */}
        <HintRevealButton
          hintBudgetRemaining={hintBudgetRemaining}
          currentHintLevel={currentHintLevel}
          onRequestHint={onRequestHint}
          isLoading={isThinking}
        />
      </div>

      {/* Messages Scroll Area */}
      <div className="flex-1 overflow-y-auto p-5 space-y-2">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center p-6 text-slate-400">
            <div className="w-12 h-12 rounded-2xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center mb-3">
              <Brain className="w-6 h-6 text-blue-400" />
            </div>
            <p className="text-sm font-semibold text-slate-200 mb-1">Interactive Socratic Session Ready</p>
            <p className="text-xs text-slate-400 max-w-xs leading-relaxed">
              Ask a question, propose a derivation, or click a concept node in the constellation to begin.
            </p>
          </div>
        ) : (
          messages.map((msg, idx) => (
            <ChatMessageItem key={msg.id || idx} message={msg} index={idx} />
          ))
        )}

        {/* Thinking Indicator */}
        {isThinking && (
          <div className="flex justify-start mb-3 animate-pulse">
            <div className="glass-card rounded-2xl px-4 py-2.5 flex items-center gap-2 border border-blue-500/30 text-xs text-blue-300 font-mono">
              <span className="w-2 h-2 rounded-full bg-blue-400 animate-ping" />
              <span>Aether is formulating Socratic guidance...</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Form Footer */}
      <div className="p-4 border-t border-slate-800/80 bg-slate-950/60">
        <form onSubmit={handleSend} className="relative flex items-center gap-2">
          <input
            type="text"
            value={inputVal}
            onChange={(e) => setInputVal(e.target.value)}
            placeholder="Type your answer, derivation, or question..."
            disabled={isThinking}
            className="flex-1 bg-slate-900/90 border border-slate-700/70 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 rounded-xl px-4 py-3 text-sm text-slate-100 placeholder-slate-500 outline-none transition-all duration-200"
          />
          <button
            type="submit"
            disabled={!inputVal.trim() || isThinking}
            className="flex items-center justify-center p-3 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:hover:bg-blue-600 text-white font-medium shadow-lg transition-all duration-200 active:scale-95"
          >
            <Send className="w-4 h-4" />
          </button>
        </form>
      </div>
    </div>
  );
}

export default ChatPanel;
