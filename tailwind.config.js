/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        surface: {
          DEFAULT: "#0b1326",
          dim: "#0b1326",
          bright: "#31394d",
          lowest: "#060e20",
          low: "#131b2e",
          container: "#171f33",
          high: "#222a3d",
          highest: "#2d3449",
          variant: "#2d3449",
        },
        primary: {
          DEFAULT: "#adc6ff",
          container: "#4d8eff",
          fixed: "#d8e2ff",
          dim: "#adc6ff",
          on: "#002e6a",
          "on-container": "#00285d",
        },
        secondary: {
          DEFAULT: "#a4c9ff",
          container: "#0267b8",
          fixed: "#d4e3ff",
          on: "#00315d",
          "on-container": "#d6e5ff",
        },
        tertiary: {
          DEFAULT: "#ffb786",
          container: "#df7412",
          on: "#502400",
          "on-container": "#461f00",
        },
        error: {
          DEFAULT: "#ffb4ab",
          container: "#93000a",
          on: "#690005",
          "on-container": "#ffdad6",
        },
        outline: {
          DEFAULT: "#8c909f",
          variant: "#424754",
        },
        "on-surface": {
          DEFAULT: "#dae2fd",
          variant: "#c2c6d6",
        },
        background: "#0b1326",
      },
      fontFamily: {
        sans: ["Inter", "sans-serif"],
        display: ["Inter", "sans-serif"],
        body: ["Inter", "sans-serif"],
        label: ["Geist", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      borderRadius: {
        DEFAULT: "0.25rem",
        lg: "0.5rem",
        xl: "0.75rem",
        "2xl": "1rem",
        full: "9999px",
      },
      spacing: {
        xs: "4px",
        sm: "8px",
        md: "16px",
        lg: "24px",
        xl: "40px",
        gutter: "24px",
        "container-max": "1280px",
      },
      boxShadow: {
        glow: "0 0 20px rgba(173, 198, 255, 0.25)",
        "active-glow": "0 0 15px rgba(173, 198, 255, 0.15)",
        "pulse-blue": "0 0 25px rgba(77, 142, 255, 0.3)",
      },
    },
  },
  plugins: [],
};
