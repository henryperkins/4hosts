Four Hosts Design Policy
========================

This policy codifies the visual and interaction language of the Four Hosts research application. It aligns existing implementation details with medium-term UX improvements so designers and engineers can iterate consistently.

Brand Personality
-----------------

- **Narrative**: Lean into the Four Hosts mythology—each paradigm embodies a distinct perspective (Dolores, Teddy, Bernard, Maeve). Use color, iconography, and copy to reinforce their roles without overwhelming the analytic core.
- **Tone**: Confident, investigative, optimistic about AI-assisted truth seeking. Messaging should balance clarity with a hint of futurism.
- **Voice**: Use concise, action-oriented language. Promote collaboration (“Let’s investigate…”) and learning (“Here’s what we uncovered”).

Visual System
-------------

### Color

- **Core Surfaces**: Continue using CSS variables (`--surface`, `--surface-subtle`, `--surface-muted`) for light/dark parity. Keep surfaces neutral to let paradigm accents stand out.
- **Paradigm Palette**: Maintain the four paradigm hues as the primary accent range. Incorporate subtle tints (`paradigm-bg-*`) for backgrounds, vibrant fills for chips/badges, and gradients for emphatic moments (headers, CTAs).
- **Semantic Colors**: Retain `--primary`, `--success`, `--warning`, `--error` for system feedback. Ensure contrast compliance in both themes; prefer 4.5:1 minimum for text on tinted fills.
- **Future Enhancements**: Extend gradients into data visualization (e.g., chart legends, metric cards) while keeping readability the priority.

### Typography

- **Font Strategy**: Adopt a dedicated font pair to replace the system stack—e.g., a geometric sans (headline) plus a humanist sans (body) imported via `@font-face`. Apply typography through Tailwind config where possible to keep components consistent.
- **Hierarchy**: Use responsive utilities (`text-responsive-*`) for major headings and clamp-based sizes for hero text. Limit bold weight to key metrics and section titles.
- **Readability**: Maintain generous line height (1.5) for body copy. Avoid all-caps except in labels/chips.

### Layout & Spacing

- **Grid**: Continue using max-width constraints (`max-w-7xl`, `max-w-4xl`) with responsive padding. Apply 24px/32px spacing rhythm on desktop, tightening to 16px on mobile.
- **Components**: Reuse card/button/input recipes defined in `index.css`. Introduce “compact” variants for dense data (e.g., history list) to avoid visual fatigue.
- **Progressive Disclosure**: Break up large panels (metrics, results) into collapsible sections or tabs using existing animation utilities to reduce scanning effort.

### Iconography & Imagery

- **Icons**: Continue using Feather icons (`react-icons/fi`). Pair each paradigm with its canonical icon (shield, heart, CPU, trend). Use filled or outlined variants consistently: outlines for navigation, filled backgrounds for status badges.
- **Illustrations**: When needed, prefer minimal line illustrations that echo the paradigms’ shapes/colors.

Motion & Interaction
--------------------

- **Timing**: Standardize easing at 200–300 ms for micro-interactions. Define `transition` helpers (fade, slide, scale) through Tailwind to avoid ad hoc values.
- **Feedback**: Ensure loading states exist for every blocking action (submit, export). Use skeletons for content shells; spinners for short operations.
- **Reduced Motion**: Respect `prefers-reduced-motion` across animations (already honored in `index.css`). Offer toggle for background animations if new effects are introduced.
- **Page Transitions**: Continue using `PageTransition` to bring coherence to route changes. Limit to fade or slide to avoid cognitive overload.

Data Visualization
------------------

- **Charts**: Map paradigms to their hex colors (see `constants/paradigm.ts`). Reserve semantic colors for statistical states (success vs. warning) to prevent confusion.
- **Density**: Provide tooltips or detail panes for complex graphs; default views should highlight key insights with plain-language summaries.
- **Responsive Behavior**: Collapse multi-column charts into stacked layouts on smaller viewports. Ensure labels remain legible by switching to abbreviations or inline legends when space is constrained.

Feedback & Messaging
--------------------

- **Toasts**: Mirror the active theme and include concise action copy. Follow success/error semantics and auto-dismiss after 4–6 seconds.
- **System Alerts**: Use the `Alert` component for non-ephemeral feedback, pairing iconography with short titles. Consider inline confirmation banners for successful form submissions.
- **User Input**: Reinforce haptic cues—focus rings, subtle lifts, and paradigm-colored glows for primary actions.
- **Contribution Loop**: After feedback submission (`AnswerFeedback`, `ClassificationFeedback`), surface a mini-summary (e.g., “We use your insights to recalibrate future answers”) to boost perceived impact.

Accessibility
-------------

- **Contrast**: Audit gradient and paradigm treatments under both themes to ensure WCAG AA contrast. Adjust tint levels where necessary.
- **Keyboard**: Maintain clear focus styles (`:focus-visible`) on interactive elements. Confirm skip links remain functional as navigation evolves.
- **Assistive Tech**: Expand `aria-live` regions for research progress updates; announce phase changes and key milestones.
- **Motion Sensitivity**: Keep the global reduced-motion override and avoid autoplaying long-running animations.

Implementation Notes
--------------------

- Keep tokens in `src/index.css` as the single source of truth. When adding new utilities, prefer Tailwind extensions over one-off classes.
- Use the theme store (`themeStore.ts`) to persist light/dark preferences. Future variations (e.g., high-contrast mode) should hook into the same pattern.
- Document component-specific variations (buttons, cards) in Storybook or similar to make the design system discoverable by engineering and QA.

Component Health Review (Q1 2024)
---------------------------------

This analysis captures the current implementation strengths and gaps across key front-end components. Use it to prioritize polish work without losing sight of the elements that already function well.

### ResearchFormEnhanced.tsx

- **Issues**: Commented `ParadigmOption` block remains in the file (lines 69-96); submitting the form never shows a classification-in-progress state; `paradigm_override` is omitted when the user leaves the override at `auto`, though the backend may expect an explicit value.
- **Strengths**: User preference hydration works reliably; query validation prevents noisy submissions; form labels and errors meet accessibility guidelines.

### ResearchProgress.tsx

- **Issues**: The `VITE_RESULTS_POLL_TIMEOUT_MS` environment flag is undocumented; the soft-timeout message tells users runs continue even after polling stops; WebSocket updates cannot be paused when the user switches tabs.
- **Strengths**: Numeric inputs are thoroughly validated; small API result sets compare efficiently; reconnection logic is resilient.

### ResearchPage.tsx

- **Issues**: Polling relies on manual error handling instead of React Query's built-ins; `getStatus` is recreated every render; two `ClassificationFeedback` instances create redundant UI.
- **Strengths**: Local state is purposeful; motion respects `prefers-reduced-motion`; polling halts cleanly when requested.

### ResultsDisplayEnhanced.tsx

- **Issues**: Metadata typing uses `Record<string, unknown>` which erodes type safety; exports lack a loading affordance; malformed backend payloads are not wrapped in an error boundary.
- **Strengths**: Optional typing guards against partial responses; panel sections collapse intuitively; export flows function end to end.

### ResearchResultPage.tsx

- **Issues**: Polling may continue after unmount when cleanup fails; inline errors never redirect to the dedicated error surface; 404 and 500 responses look identical.
- **Strengths**: The two-second polling cadence balances freshness and load; polling stops promptly on completion; unmount cleanup typically succeeds.

### ContextMetricsPanel.tsx

- **Issues**: Sub-10 ms operations render as `0.00s` instead of `<0.01s`; individual state setters fire sequentially, introducing extra renders; panel lacks a manual refresh.
- **Strengths**: Responsive grid adapts across breakpoints; acronym legend clarifies terminology; loading and error states are well defined.

### EvidencePanel.tsx

- **Issues**: Fallback keys still rely on the array index; timestamp formatting ignores the user's locale; `slice(0, maxInitial)` runs on every render.
- **Strengths**: Primary key generation supports Unicode content; credibility indicators expose ARIA metadata; expand/collapse interactions feel smooth.

### Navigation.tsx

- **Issues**: The "4H Research" abbreviation can confuse new users; logging out does not close WebSocket sessions before redirecting; `getParadigmHoverClass` re-instantiates each render.
- **Strengths**: Breakpoints are deliberate; skip link remains functional; mobile transitions are fluid.

#### Cross-Cutting Priorities

- **Performance**: Extract render-stable helpers in `ResearchPage` and `Navigation`, and batch state updates in `ContextMetricsPanel`.
- **UX**: Remove stale comments from `ResearchFormEnhanced`, clarify the `ResearchProgress` timeout message, and consolidate the dual feedback panels in `ResearchPage`.
- **Data Handling**: Replace broad `Record<string, unknown>` metadata typing, localize evidence timestamps, and differentiate error statuses in `ResearchResultPage`.
- **Missing Enhancements**: Add a metrics refresh trigger, allow pausing research updates during tab switches, and tear down live connections on logout.

Roadmap Items
-------------

1. Integrate new font stack and update Tailwind typography plugin settings.
2. Refresh metrics cards with paradigm-infused gradients and optional compact mode.
3. Add collapsible sections to `ResultsDisplayEnhanced` and `ResearchProgress` for improved scannability.
4. Extend skeleton loaders to research submission and history views.
5. Implement aria-live announcements for progress phases and feedback confirmations.

By following this policy, the application maintains its distinctive Four Hosts narrative while ensuring usability, accessibility, and visual cohesion across rapid product iterations.
