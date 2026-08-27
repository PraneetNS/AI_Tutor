/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        background: "#0a0a0f",
        card: "rgba(18, 18, 26, 0.75)",
        gold: {
          DEFAULT: "#f0b429",
          glow: "rgba(240, 180, 41, 0.4)",
          dark: "#b7791f",
        },
        blue: {
          DEFAULT: "#3b82f6",
          glow: "rgba(59, 130, 246, 0.4)",
          dark: "#1d4ed8",
        },
        locked: {
          DEFAULT: "#4b4b55",
          dark: "#2a2a32",
        },
      },
      fontFamily: {
        sans: ['"Plus Jakarta Sans"', 'Inter', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
      animation: {
        'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'pulse-fast': 'pulse 1.8s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
    },
  },
  plugins: [],
}
