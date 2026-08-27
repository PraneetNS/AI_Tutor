/**
 * Initial Machine Learning Curriculum Concept Graph.
 * Matches backend ML_CONCEPTS & ML_EDGES with deterministic 3D layout coordinates.
 */

export const INITIAL_CONCEPTS = [
  // Foundations (Mastered)
  {
    id: "variables",
    name: "Variables & Data Types",
    domain: "programming",
    status: "mastered",
    mastery: 0.95,
    position: [-10, 4, -4],
  },
  {
    id: "expressions",
    name: "Expressions & Operators",
    domain: "programming",
    status: "mastered",
    mastery: 0.92,
    position: [-8, 7, -6],
  },
  {
    id: "functions",
    name: "Functions & Scope",
    domain: "programming",
    status: "mastered",
    mastery: 0.88,
    position: [-11, 0, -2],
  },
  {
    id: "linear_algebra",
    name: "Linear Algebra Basics",
    domain: "mathematics",
    status: "mastered",
    mastery: 0.89,
    position: [-6, 3, -1],
  },
  {
    id: "calculus_basics",
    name: "Calculus Basics",
    domain: "mathematics",
    status: "mastered",
    mastery: 0.85,
    position: [-5, -4, -3],
  },
  {
    id: "chain_rule",
    name: "Chain Rule",
    domain: "mathematics",
    status: "mastered",
    mastery: 0.82,
    position: [-2, -5, -1],
  },
  {
    id: "partial_derivatives",
    name: "Partial Derivatives",
    domain: "mathematics",
    status: "mastered",
    mastery: 0.80,
    position: [-3, -1, 1],
  },
  {
    id: "probability",
    name: "Probability & Statistics",
    domain: "mathematics",
    status: "mastered",
    mastery: 0.78,
    position: [-7, -2, -6],
  },

  // Active / In Progress (ML Core)
  {
    id: "supervised_learning",
    name: "Supervised Learning",
    domain: "machine_learning",
    status: "in_progress",
    mastery: 0.65,
    position: [-1, 3, 2],
  },
  {
    id: "loss_functions",
    name: "Loss Functions",
    domain: "machine_learning",
    status: "in_progress",
    mastery: 0.58,
    position: [2, 4, -1],
  },
  {
    id: "gradient_descent",
    name: "Gradient Descent",
    domain: "machine_learning",
    status: "in_progress",
    mastery: 0.52,
    position: [3, 0, 3],
  },
  {
    id: "backpropagation",
    name: "Backpropagation",
    domain: "machine_learning",
    status: "in_progress",
    mastery: 0.44,
    position: [4, -3, 0],
  },

  // Advanced / Locked
  {
    id: "neural_networks",
    name: "Neural Networks",
    domain: "machine_learning",
    status: "locked",
    mastery: 0.15,
    position: [7, -1, 2],
  },
  {
    id: "regularization",
    name: "Regularization (L1/L2)",
    domain: "machine_learning",
    status: "locked",
    mastery: 0.10,
    position: [6, 5, -3],
  },
  {
    id: "gradient_descent_variants",
    name: "Optimizer Variants (Adam)",
    domain: "machine_learning",
    status: "locked",
    mastery: 0.05,
    position: [8, 2, -1],
  },
  {
    id: "attention_mechanisms",
    name: "Attention Mechanisms",
    domain: "machine_learning",
    status: "locked",
    mastery: 0.0,
    position: [11, -3, -2],
  },
  {
    id: "transformers",
    name: "Transformer Architecture",
    domain: "machine_learning",
    status: "locked",
    mastery: 0.0,
    position: [13, 0, 1],
  },
];

export const INITIAL_EDGES = [
  // Math & CS
  { source: "calculus_basics", target: "chain_rule" },
  { source: "calculus_basics", target: "partial_derivatives" },
  { source: "linear_algebra", target: "partial_derivatives" },
  { source: "linear_algebra", target: "supervised_learning" },
  { source: "probability", target: "supervised_learning" },
  { source: "supervised_learning", target: "loss_functions" },
  { source: "probability", target: "loss_functions" },

  // Gradient Descent & Backpropagation
  { source: "loss_functions", target: "gradient_descent" },
  { source: "partial_derivatives", target: "gradient_descent" },
  { source: "gradient_descent", target: "backpropagation" },
  { source: "chain_rule", target: "backpropagation" },
  { source: "partial_derivatives", target: "backpropagation" },

  // Neural Networks & Beyond
  { source: "backpropagation", target: "neural_networks" },
  { source: "loss_functions", target: "neural_networks" },
  { source: "neural_networks", target: "regularization" },
  { source: "gradient_descent", target: "regularization" },
  { source: "gradient_descent", target: "gradient_descent_variants" },
  { source: "neural_networks", target: "attention_mechanisms" },
  { source: "linear_algebra", target: "attention_mechanisms" },
  { source: "attention_mechanisms", target: "transformers" },
  { source: "regularization", target: "transformers" },
];
