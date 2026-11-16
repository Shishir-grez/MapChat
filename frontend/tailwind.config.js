import forms from '@tailwindcss/forms';

/**
 * Design tokens are CSS variables (space-separated RGB channels) so each UI variant can
 * re-theme the shared utilities: set --mc-* on the variant's root element.
 * @type {import('tailwindcss').Config}
 */
const token = (name) => `rgb(var(--mc-${name}) / <alpha-value>)`;

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        canvas: token('canvas'),
        surface: token('surface'),
        ink: token('ink'),
        muted: token('muted'),
        faint: token('faint'),
        line: token('line'),
        accent: token('accent'),
        'accent-ink': token('accent-ink'),
      },
      fontFamily: {
        display: 'var(--mc-font-display)',
        sans: 'var(--mc-font-sans)',
        mono: 'var(--mc-font-mono)',
      },
    },
  },
  // 'class' strategy: only elements with form-* classes get the plugin's resets.
  plugins: [forms({ strategy: 'class' })],
};
