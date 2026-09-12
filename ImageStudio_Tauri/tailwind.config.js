/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#090C12",
        panel: {
          DEFAULT: "#0F131B",
          elevated: "#151A24",
          hover: "#1D2433",
        },
        border: {
          DEFAULT: "#252C3A",
          hover: "#353F54",
          focus: "#7C6CFF",
        },
        brand: {
          primary: "#7C6CFF",
          secondary: "#4F9CFF",
          accent: "#35D6C5",
          amber: "#F59E0B",
          danger: "#FF5C6C",
          success: "#10B981",
        },
        text: {
          primary: "#E8ECF4",
          muted: "#8993A7",
          disabled: "#555E70",
        },
      },
      fontFamily: {
        sans: ["Segoe UI", "Plus Jakarta Sans", "-apple-system", "sans-serif"],
        mono: ["JetBrains Mono", "Consolas", "monospace"],
      },
      borderRadius: {
        card: "12px",
        control: "8px",
        button: "10px",
      },
    },
  },
  plugins: [],
}
