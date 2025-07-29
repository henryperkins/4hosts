# Tailwind v4 Migration Playbook

Concrete, sequential instructions for aligning the current **four-hosts-app** frontend with recommendations from the *Tailwind CSS v4 Quick Reference Guide*.

---

## 0. Reading guide

* **Perform steps in order** – every heading is safe to commit individually.
* Commands are illustrative.  Adapt paths if you keep a different folder layout.
* When you see «copy the value», grab exactly the numbers you are already using – no design changes are introduced by this script.

---

## 1 Replace the inert `@theme extend` block

### 1-a Create `tailwind.config.ts` (preferred)

```ts
// tailwind.config.ts
import type { Config } from 'tailwindcss'

export default <Partial<Config>>{
  theme: {
    extend: {
      colors: {
        paradigm: {
          dolores: 'oklch(0.629 0.237 24.8)',
          teddy:   'oklch(0.7   0.193 42.7)',
          bernard: 'oklch(0.598 0.216 252.1)',
          maeve:   'oklch(0.671 0.186 157.6)',
        },
        surface: {
          DEFAULT: 'oklch(1 0 0)',
          subtle : 'oklch(0.98 0.005 240)',
          muted  : 'oklch(0.95 0.01 240)',
        },
      },
      borderRadius: {
        sm : '0.25rem',
        md : '0.375rem',
        lg : '0.5rem',
        xl : '0.75rem',
        '2xl': '1rem',
        full: '9999px',
      },
      // shadows, easing, keyframes – copy existing values
    },
  },
  plugins: [],
}
```

### 1-b Delete the `@theme extend { … }` section

Remove it from `src/app.css` (or wherever it currently lives).

### 1-c Keep Tailwind import

The CSS file must still start with:

```css
@import "tailwindcss";
```

*(If you insist on CSS-only configuration: rename the block to plain `@theme { … }` instead of deleting, but remember the values must be valid CSS, not commented pseudo-syntax.)*

---

## 2 Normalise CSS variable naming

### 2-a Global replace

```
--color-paradigm- → --paradigm-
--color-surface   → --surface
```

### 2-b Keep an alias bridge (optional)

```css
:root {
  --color-paradigm-dolores: var(--paradigm-dolores);
  /* repeat for teddy, bernard, maeve */
}
```

---

## 3 Re-build gradients & fall-backs from variables

### 3-a Edit `src/styles/components.css`

```css
.paradigm-bg-dolores  { background: linear-gradient(to br, var(--paradigm-dolores)/10%, var(--paradigm-dolores)/4%); }
.paradigm-bg-teddy    { background: linear-gradient(to br, var(--paradigm-teddy)/10%,   var(--paradigm-teddy)/4%); }
.paradigm-bg-bernard  { background: linear-gradient(to br, var(--paradigm-bernard)/10%, var(--paradigm-bernard)/4%); }
.paradigm-bg-maeve    { background: linear-gradient(to br, var(--paradigm-maeve)/10%,   var(--paradigm-maeve)/4%); }

@media (prefers-color-scheme: dark) {
  .dark .paradigm-bg-dolores { background: linear-gradient(to br, var(--paradigm-dolores)/20%, var(--paradigm-dolores)/10%); }
  /* repeat for the other paradigms */
}
```

### 3-b Delete old rgb() gradients

Remove the hard-coded colour blocks you just replaced.

### 3-c Layer the fall-back file

```css
@layer theme {
  @supports not (color: oklch(1 0 0)) {
    :root { /* your HSL values */ }
  }
}
```

---

## 4 Consolidate Button styles

### 4-a Delete duplicate `.btn-primary` block

Erase it from `components.css` – we will re-add, but smaller.

### 4-b Adjust React component

```tsx
const variantClasses = {
  primary: 'btn-primary',
  /* other variants unchanged */
}
```

### 4-c Re-declare once via @apply

```css
@layer components {
  .btn-primary {
    @apply bg-blue-600 text-white hover:bg-blue-700
           focus-visible:ring-2 focus-visible:ring-blue-500
           transition-all duration-200 rounded-md;
  }
}
```

---

## 5 Fix InputField baseline & missing animation

### 5-a Add baseline class

```css
@layer components {
  .input-field {
    @apply block w-full border border-border rounded-md px-3 py-2
           text-text bg-surface placeholder:text-text-muted
           focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500;
  }
}
```

### 5-b Define `pulse-border` keyframes

```css
@keyframes pulse-border {
  0%,100% { box-shadow: 0 0 0 0 var(--tw-ring-color); }
  50%     { box-shadow: 0 0 0 4px transparent; }
}

.animate-pulse-border { animation: pulse-border 1.5s ease-in-out infinite; }
```

---

## 6 Move paradigm helpers to `@utility`

### 6-a Create `src/styles/paradigm.utilities.css`

```css
@import "tailwindcss";

@utility text-paradigm-dolores   { color: var(--paradigm-dolores); }
@utility bg-paradigm-dolores     { background-color: var(--paradigm-dolores); }
@utility border-paradigm-dolores { border-color: var(--paradigm-dolores); }
/* repeat for teddy, bernard, maeve */
```

### 6-b Refactor JSX usages

Replace arbitrary utilities:

```
text-[--paradigm-dolores]  → text-paradigm-dolores
bg-[--paradigm-dolores]    → bg-paradigm-dolores
border-[--paradigm-dolores]→ border-paradigm-dolores
```

### 6-c Delete old manual borders

Remove `.paradigm-border-*` from `components.css` (now redundant).

---

## 7 Textarea: native auto-size

1. Delete the JS `textareaRef.current` height logic in `ResearchForm.tsx`.
2. Add `className="field-sizing-content resize-y"` to `<InputField …>`.
3. Ensure `.input-field` baseline no longer forces `resize-none`.

---

## 8 Unify disabled / aria logic

```tsx
const isDisabled = query.trim().length < 10 || isLoading;

<Button
  type="submit"
  disabled={isDisabled}
  aria-disabled={isDisabled}
  …
/>
```

Remove duplicate checks **inside** the Button component so it relies on the prop.

---

## 9 Purge safety & content scanning

### 9-a Tell Tailwind to scan markdown/docs

```css
@source "./docs/**/*.md";
```

### 9-b Temporary safelist in `tailwind.config.ts`

```ts
export default <Partial<Config>>{
  safelist: [
    { pattern: /(text|bg|border)-paradigm-(dolores|teddy|bernard|maeve)/ },
    { pattern: /paradigm-bg-(dolores|teddy|bernard|maeve)/ },
  ],
}
```

---

## 10 Colour-contrast QA

1. `npm i -D @tailwindcss/a11y` and enable the plugin.
2. Run the audit; darken Teddy & Dolores text colours (`l – 0.05` in OKLCH) until AA passes on white backgrounds.

---

### ✅  You’re done!

The app will now:

* use Tailwind’s v4-native configuration,
* avoid duplicate CSS and RGB drift,
* keep all paradigm utilities purge-safe,
* and honour accessibility + modern variants.
