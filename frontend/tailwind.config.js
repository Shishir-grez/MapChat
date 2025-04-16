/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        blue: {
          50: '#E3EFFF',
          100: '#D4E4FF',
          500: '#3B7DFF',
          600: '#0052CC',
          700: '#0747A6',
        },
        gray: {
          50: '#FAFAFA',
          100: '#F5F5F5',
          200: '#E0E0E0',
          300: '#CCCCCC',
          400: '#AAAAAA',
          500: '#888888',
          600: '#666666',
          700: '#444444',
          800: '#333333',
          900: '#222222',
        },
      },
      boxShadow: {
        sm: '0 1px 3px rgba(0,0,0,0.12)',
        md: '0 0 20px rgba(0,0,0,0.08)',
        lg: '0 10px 30px rgba(0,0,0,0.15)',
      },
      keyframes: {
        messageIn: {
          'from': { opacity: 0, transform: 'translateY(10px)' },
          'to': { opacity: 1, transform: 'translateY(0)' }
        },
        typingDot: {
          '0%, 100%': { opacity: 0.2, transform: 'translateY(0)' },
          '50%': { opacity: 1, transform: 'translateY(-2px)' }
        },
        'subtle-pulse': {
          '0%': { transform: 'scale(1)', opacity: 0.8 },
          '50%': { transform: 'scale(1.05)', opacity: 1 },
          '100%': { transform: 'scale(1)', opacity: 0.8 }
        }
      },
      animation: {
        messageIn: 'messageIn 0.3s ease-out',
        typingDot: 'typingDot 0.9s infinite',
        'subtle-pulse': 'subtle-pulse 1.5s ease-in-out infinite'
      },
      backdropFilter: {
        'none': 'none',
        'blur': 'blur(10px)',
      },
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
  ],
};
