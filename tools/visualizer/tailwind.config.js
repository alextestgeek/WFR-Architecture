/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        display: ['Clash Display', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      colors: {
        primary: {
          50: '#f0f9ff',
          500: '#22d3ee',
          600: '#06b6d4',
        },
        accent: {
          500: '#c026d3',
          600: '#a21caf',
        },
        surface: '#0a0a0a',
        'surface-2': '#111111',
      },
      animation: {
        'wave': 'wave 4s cubic-bezier(0.4, 0, 0.2, 1) infinite',
        'resonate': 'resonate 2.5s ease-in-out infinite alternate',
      }
    },
  },
  plugins: [],
}
