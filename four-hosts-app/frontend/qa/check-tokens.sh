#!/usr/bin/env bash
set -euo pipefail

# Directories to scan
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Targeted files (scope this check to the recently migrated components)
FILES=(
  "$ROOT/src/components/ResultsDisplayEnhanced.tsx"
  "$ROOT/src/components/UserProfile.tsx"
  "$ROOT/src/components/ResearchResultPage.tsx"
  "$ROOT/src/components/Navigation.tsx"
  "$ROOT/src/components/MetricsDashboard.tsx"
  "$ROOT/src/components/ResearchProgress.tsx"
  "$ROOT/src/components/ParadigmDisplay.tsx"
  "$ROOT/src/components/ErrorBoundary.tsx"
)

# Patterns that must not appear (raw Tailwind palette usage)
PATTERNS=(
  "bg-white"
  "bg-gray-"
  "text-gray-"
  "border-gray-"
  "dark:bg-gray-"
  "dark:text-gray-"
  "text-blue-"
  "dark:text-blue-"
  "from-blue-"
  "to-blue-"
)

FAIL=0
for p in "${PATTERNS[@]}"; do
  if grep -RIn --label="$p" "$p" "${FILES[@]}" >/dev/null 2>&1; then
    echo "Design token violation: pattern '$p' found in components (exclude examples)." >&2
    grep -RIn "$p" "${FILES[@]}" || true
    FAIL=1
  fi
done

if [ $FAIL -ne 0 ]; then
  cat <<EOF >&2

Use tokenized classes instead:
  - bg-surface, bg-surface-subtle, bg-surface-muted
  - text-text, text-text-muted, text-text-subtle, text-primary
  - border-border
  - color semantics: text-primary/bg-primary, text-success/bg-success, text-error/bg-error

You can re-run this check via: npm run design:lint
EOF
  exit 1
fi

echo "Design token check passed."
