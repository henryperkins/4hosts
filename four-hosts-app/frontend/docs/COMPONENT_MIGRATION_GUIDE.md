# Component Migration Guide: Tailwind v4 & Design System Alignment

This guide provides step-by-step instructions for updating existing components to align with the new design system, Tailwind v4 patterns, and accessibility standards implemented in ResearchFormEnhanced and SkeletonLoader.

## Table of Contents
1. [Core Principles](#core-principles)
2. [Color System Migration](#color-system-migration)
3. [Component Structure](#component-structure)
4. [Accessibility Improvements](#accessibility-improvements)
5. [Animation & Motion](#animation--motion)
6. [Common Patterns](#common-patterns)
7. [Migration Checklist](#migration-checklist)

## Core Principles

### 1. CSS Variables Over Hard-coded Values
- ❌ **Old**: `bg-gray-800`, `text-gray-100`, `border-gray-200`
- ✅ **New**: `bg-surface`, `text-text`, `border-border`

### 2. Proper CSS Layering
- ❌ **Old**: Inline styles or unlayered CSS
- ✅ **New**: Use `@layer components` or `@layer utilities`

### 3. Focus on Accessibility
- ❌ **Old**: Visual-only indicators
- ✅ **New**: Proper ARIA attributes, roles, and keyboard support

### 4. Consistent Animation Patterns
- ❌ **Old**: Inline `style={{ animationDelay }}`
- ✅ **New**: Animation utility classes with reduced motion support

## Color System Migration

### Semantic Color Variables

Replace hard-coded Tailwind colors with semantic CSS variables:

```tsx
// ❌ OLD
<div className="bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100">
  <p className="text-gray-600 dark:text-gray-400">Muted text</p>
</div>

// ✅ NEW
<div className="bg-surface text-text">
  <p className="text-text-muted">Muted text</p>
</div>
```

### Color Variable Reference
```css
/* Surfaces */
--color-surface        /* Primary background (white/dark) */
--color-surface-subtle /* Secondary background */
--color-surface-muted  /* Tertiary background */

/* Borders */
--color-border        /* Primary borders */
--color-border-subtle /* Subtle borders */

/* Text */
--color-text         /* Primary text */
--color-text-muted   /* Secondary text */

/* Paradigm Colors */
--color-paradigm-dolores
--color-paradigm-teddy
--color-paradigm-bernard
--color-paradigm-maeve
```

### Paradigm Colors

```tsx
// ❌ OLD
<div className={`bg-${paradigm}-500 text-white`}>
<div className="bg-red-100 text-red-800 border-red-200">

// ✅ NEW
<div className={`bg-[--color-paradigm-${paradigm}] text-white`}>
<div className="paradigm-bg-dolores text-[--color-paradigm-dolores]">
```

## Component Structure

### Card Components

Replace manual card implementations with the Card component:

```tsx
// ❌ OLD
<div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
  <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Title</h3>
  <p className="text-gray-600 dark:text-gray-400">Content</p>
</div>

// ✅ NEW
import { Card, CardHeader, CardTitle, CardContent } from './ui/Card'

<Card>
  <CardHeader>
    <CardTitle>Title</CardTitle>
  </CardHeader>
  <CardContent>Content</CardContent>
</Card>
```

### Button Components

```tsx
// ❌ OLD
<button className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700">
  Click me
</button>

// ✅ NEW
import { Button } from './ui/Button'

<Button variant="primary" onClick={handleClick}>
  Click me
</Button>
```

### Form Inputs

```tsx
// ❌ OLD
<input
  type="text"
  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
  placeholder="Enter text"
/>

// ✅ NEW
import { InputField } from './ui/InputField'

<InputField
  label="Field Label"
  placeholder="Enter text"
  status={error ? 'error' : undefined}
  errorMessage={error}
  hint="Helper text"
/>
```

## Accessibility Improvements

### ARIA Attributes

```tsx
// ❌ OLD
<div className="loading-spinner"></div>
<button disabled={loading}>Submit</button>

// ✅ NEW
<div className="loading-spinner" role="progressbar" aria-label="Loading"></div>
<button disabled={loading} aria-busy={loading} aria-disabled={loading}>
  Submit
</button>
```

### Focus Management

```tsx
// ❌ OLD
<button className="focus:outline-none focus:ring-2">

// ✅ NEW - Only show focus ring for keyboard navigation
<button className="focus-visible:outline-none focus-visible:ring-2">
```

### Status Announcements

```tsx
// ❌ OLD
{error && <div className="text-red-500">{error}</div>}

// ✅ NEW
{error && (
  <div role="alert" aria-live="polite" className="text-red-500">
    {error}
  </div>
)}
```

## Animation & Motion

### Animation Classes

```tsx
// ❌ OLD
<div style={{ animationDelay: `${index * 100}ms` }} className="animate-fade-in">

// ✅ NEW - Use CSS custom properties or nth-child
<div className="animate-fade-in stagger-animation">
```

### Reduced Motion Support

All animations should respect user preferences:

```css
/* Automatically handled in animations.css */
@media (prefers-reduced-motion: reduce) {
  .animate-* {
    animation: none !important;
  }
}
```

### Hover Effects

```tsx
// ❌ OLD
<div className="hover:shadow-lg hover:scale-105 hover:-translate-y-1">

// ✅ NEW - Use consistent hover patterns
<div className="card-interactive"> /* or */ <div className="hover-lift">
```

## Common Patterns

### Status Indicators

```tsx
// ❌ OLD
<div className="flex items-center gap-2">
  {status === 'success' && <CheckCircle className="h-4 w-4 text-green-500" />}
  {status === 'error' && <XCircle className="h-4 w-4 text-red-500" />}
  <span>{status}</span>
</div>

// ✅ NEW - Use StatusIcon component (when created)
<StatusIcon status={status} showLabel />
```

### Loading States

```tsx
// ❌ OLD
<div className="animate-spin">
  <Loader className="h-6 w-6" />
</div>

// ✅ NEW
<span className="loading-spinner" role="progressbar" aria-label="Loading">
  <span className="sr-only">Loading...</span>
</span>
```

### Badges

```tsx
// ❌ OLD
<span className="px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-sm">
  Active
</span>

// ✅ NEW
import { Badge } from './ui/Badge'

<Badge variant="info">Active</Badge>
```

## Migration Checklist

### For Each Component:

- [ ] **Replace Colors**
  - [ ] Replace `gray-*` with semantic colors (`surface`, `text`, `border`)
  - [ ] Replace paradigm colors with CSS variables
  - [ ] Remove dark mode color duplication

- [ ] **Update Structure**
  - [ ] Replace manual cards with `Card` component
  - [ ] Replace manual buttons with `Button` component
  - [ ] Replace manual inputs with `InputField` component
  - [ ] Extract repeated patterns into components

- [ ] **Add Accessibility**
  - [ ] Add proper ARIA labels and roles
  - [ ] Add `aria-busy` for loading states
  - [ ] Add `aria-invalid` for error states
  - [ ] Use `focus-visible` instead of `focus`
  - [ ] Add keyboard navigation support

- [ ] **Fix Animations**
  - [ ] Remove inline animation delays
  - [ ] Use animation utility classes
  - [ ] Ensure reduced motion support
  - [ ] Use consistent transition timings

- [ ] **Clean Up**
  - [ ] Remove inline styles
  - [ ] Remove unused imports
  - [ ] Consolidate duplicate logic
  - [ ] Add proper TypeScript types

## Example Migration

### Before (Old Component)
```tsx
<div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 hover:shadow-lg transition-shadow duration-200">
  <div className="flex items-center justify-between mb-2">
    <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
      {title}
    </h3>
    <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-sm">
      {status}
    </span>
  </div>
  <p className="text-gray-600 dark:text-gray-400">{description}</p>
  <button 
    className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
    style={{ animationDelay: '200ms' }}
  >
    View Details
  </button>
</div>
```

### After (Migrated Component)
```tsx
import { Card, CardHeader, CardTitle, CardContent } from './ui/Card'
import { Badge } from './ui/Badge'
import { Button } from './ui/Button'

<Card className="animate-fade-in">
  <CardHeader>
    <div className="flex items-center justify-between">
      <CardTitle>{title}</CardTitle>
      <Badge variant="info">{status}</Badge>
    </div>
  </CardHeader>
  <CardContent>
    <p className="text-text-muted mb-4">{description}</p>
    <Button variant="primary" onClick={handleViewDetails}>
      View Details
    </Button>
  </CardContent>
</Card>
```

## Testing Your Migration

1. **Visual Testing**
   - Component looks correct in light and dark modes
   - Hover states work properly
   - Animations are smooth

2. **Accessibility Testing**
   - Keyboard navigation works
   - Screen reader announces properly
   - Focus indicators are visible

3. **Performance Testing**
   - No layout shifts
   - Animations respect reduced motion
   - Bundle size hasn't increased significantly

## Resources

- [Tailwind v4 Migration Guide](../docs/tailwindv4guide.md)
- [UI/UX Best Practices](../docs/ui.md)
- [Animation Guidelines](../docs/animation.md)
- Component Examples: `ResearchFormEnhanced.tsx`, `SkeletonLoader.tsx`