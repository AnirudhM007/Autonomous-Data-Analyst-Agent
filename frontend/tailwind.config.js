/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        paper: "#f6f4ed",
        ink: "#181817",
        tomato: "#ff704c",
        acid: "#dafa62",
      },
    },
  },
  plugins: [],
};
