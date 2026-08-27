import { useState, useCallback, useRef } from "react";
import { INITIAL_CONCEPTS, INITIAL_EDGES } from "../mock/initialGraph";

const API_BASE = "http://localhost:8000";

/**
 * useTutorSession Hook:
 * Manages conversation history, pedagogical state, hint budget, and concept graph state.
 * Connects to the backend API with seamless local mock fallback.
 */
export function useTutorSession() {
  const [messages, setMessages] = useState([
    {
      id: "msg_init",
      role: "assistant",
      content:
        "Welcome! I am Aether, your AI Tutor for Machine Learning. We are currently exploring Gradient Descent and Backpropagation. How would you explain why we compute partial derivatives during the backward pass?",
      pedagogyMode: "socratic",
      targetConcept: "Backpropagation",
      isHint: false,
    },
  ]);

  const [pedagogyMode, setPedagogyMode] = useState("idle"); // idle | thinking | hint | celebrate | stuck
  const [hintLevel, setHintLevel] = useState(0);
  const [hintBudgetRemaining, setHintBudgetRemaining] = useState(3);
  const [targetConcept, setTargetConcept] = useState("Backpropagation");
  const [isThinking, setIsThinking] = useState(false);
  const [concepts, setConcepts] = useState(INITIAL_CONCEPTS);
  const [edges] = useState(INITIAL_EDGES);

  const constellationRef = useRef(null);

  // Send a student message to the tutor
  const sendMessage = useCallback(
    async (studentText) => {
      const userMsg = {
        id: `msg_${Date.now()}`,
        role: "user",
        content: studentText,
      };

      setMessages((prev) => [...prev, userMsg]);
      setIsThinking(true);
      setPedagogyMode("thinking");

      try {
        const res = await fetch(`${API_BASE}/api/ai/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message: studentText,
            course_id: "ml_101",
            lecture_id: 1,
            conversation_history: messages.map((m) => ({
              role: m.role === "assistant" ? "assistant" : "user",
              content: m.content,
            })),
          }),
        });

        if (res.ok) {
          const data = await res.json();
          const tutorMsg = {
            id: `msg_${Date.now() + 1}`,
            role: "assistant",
            content: data.answer,
            pedagogyMode: data.pedagogy_mode || "socratic",
            targetConcept: data.topic || targetConcept,
            hintLevel: data.hint_level || 0,
            isHint: data.pedagogy_mode === "hint" || (data.hint_level || 0) > 0,
          };

          setMessages((prev) => [...prev, tutorMsg]);
          setHintLevel(data.hint_level || 0);
          setPedagogyMode(data.pedagogy_mode?.toLowerCase() || "idle");
        } else {
          throw new Error("Backend unavailable");
        }
      } catch (err) {
        // High-Fidelity Mock Pedagogical Fallback when backend is offline
        setTimeout(() => {
          let responseText = "";
          let nextMode = "socratic";
          let isHint = false;

          const textLower = studentText.toLowerCase();

          if (textLower.includes("hint") || textLower.includes("help") || textLower.includes("stuck")) {
            nextMode = "hint";
            isHint = true;
            const nextLevel = Math.min(3, hintLevel + 1);
            setHintLevel(nextLevel);
            setHintBudgetRemaining((b) => Math.max(0, b - 1));

            if (nextLevel === 1) {
              responseText = "Think about the Chain Rule: how does a change in the weight w affect the loss L through the pre-activation z?";
            } else if (nextLevel === 2) {
              responseText = "Recall that ∂L/∂w = (∂L/∂a) * (∂a/∂z) * (∂z/∂w). Notice how each layer's local derivative multiplies backward.";
            } else {
              responseText = "Specifically: ∂z/∂w is simply the activation from the previous layer x. So the gradient is δ * x.";
            }
          } else if (textLower.includes("chain rule") || textLower.includes("derivative") || textLower.includes("multiply")) {
            nextMode = "celebrate";
            responseText =
              "Spot on! The chain rule allows us to calculate how each intermediate weight contributed to the final output error by multiplying local derivatives backward.\n\nNow, what happens if the activation function saturates (derivative near zero)?";

            // Trigger visual node transition in 3D constellation
            if (constellationRef.current) {
              constellationRef.current.updateNode("backpropagation", "mastered", 0.95);
            }
          } else {
            nextMode = "socratic";
            responseText =
              "That's an interesting perspective. If we only computed the forward pass without tracking gradients, how would the optimizer know which direction reduces the cost function?";
          }

          const tutorMsg = {
            id: `msg_${Date.now() + 1}`,
            role: "assistant",
            content: responseText,
            pedagogyMode: nextMode,
            targetConcept: targetConcept,
            hintLevel: hintLevel,
            isHint: isHint,
          };

          setMessages((prev) => [...prev, tutorMsg]);
          setPedagogyMode(nextMode);
        }, 1000);
      } finally {
        setTimeout(() => {
          setIsThinking(false);
        }, 1100);
      }
    },
    [messages, hintLevel, targetConcept]
  );

  // Explicit hint request trigger
  const requestHint = useCallback(() => {
    if (hintBudgetRemaining <= 0 || isThinking) return;
    sendMessage("Could you give me a hint for this step?");
  }, [hintBudgetRemaining, isThinking, sendMessage]);

  // Handle node selection from constellation
  const handleNodeClick = useCallback((node) => {
    setTargetConcept(node.name);
    setPedagogyMode("thinking");
    setTimeout(() => {
      setMessages((prev) => [
        ...prev,
        {
          id: `msg_${Date.now()}`,
          role: "assistant",
          content: `We are now focusing on '${node.name}' (Current status: ${node.status}, Mastery: ${Math.round(
            node.mastery * 100
          )}%). What question or calculation would you like to explore here?`,
          pedagogyMode: "socratic",
          targetConcept: node.name,
          isHint: false,
        },
      ]);
      setPedagogyMode("idle");
    }, 400);
  }, []);

  return {
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
  };
}

export default useTutorSession;
