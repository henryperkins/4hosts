#!/bin/bash

# Claude Code post-edit hook: Clean up deprecated code after file modifications
# This hook runs after Claude modifies any files

HOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HOOK_DIR/../.." && pwd)"
SCRIPTS_DIR="$HOOK_DIR/scripts"

# Create scripts directory if it doesn't exist
mkdir -p "$SCRIPTS_DIR"

echo "🧹 Running post-edit cleanup..."
echo ""
echo "⚠️  IMPORTANT REMINDER FOR CLAUDE:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "You MUST remove any code that has been replaced or deprecated by your recent"
echo "modifications. Leaving outdated, unused, or redundant code in the repository"
echo "is AGAINST THE RULES. This includes:"
echo "  • Unused imports that are no longer needed"
echo "  • Functions/classes that have been replaced by new implementations"
echo "  • Old component files superseded by new ones"
echo "  • Configuration files for removed features"
echo "  • Test files for deleted functionality"
echo ""
echo "The repository must remain lean and contain ONLY logic relevant to the current"
echo "working build. If you've created new implementations, ensure the old ones are"
echo "removed. Check for and update any dependent files as well."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Get list of modified files from git
MODIFIED_FILES=$(cd "$REPO_ROOT" && git diff --name-only HEAD 2>/dev/null || git ls-files -m 2>/dev/null)

if [ -z "$MODIFIED_FILES" ]; then
    echo "No modified files detected, skipping cleanup"
    exit 0
fi

echo "Modified files detected:"
echo "$MODIFIED_FILES"

# Run the cleanup analyzer on modified files
if [ -f "$SCRIPTS_DIR/cleanup-deprecated.py" ]; then
    echo "$MODIFIED_FILES" | python3 "$SCRIPTS_DIR/cleanup-deprecated.py" --modified-files "$REPO_ROOT"
else
    echo "⚠️  Cleanup script not found at $SCRIPTS_DIR/cleanup-deprecated.py"
fi

# Run dependency updater on files that import modified modules
if [ -f "$SCRIPTS_DIR/update-dependencies.py" ]; then
    echo "$MODIFIED_FILES" | python3 "$SCRIPTS_DIR/update-dependencies.py" --check-imports "$REPO_ROOT"
else
    echo "⚠️  Dependency updater not found at $SCRIPTS_DIR/update-dependencies.py"
fi

echo "✅ Post-edit cleanup complete"
echo ""
echo "📋 CLEANUP CHECKLIST FOR CLAUDE:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "After making changes, verify you have:"
echo "  ✓ Removed old implementations that were replaced"
echo "  ✓ Deleted unused import statements"
echo "  ✓ Removed configuration for deprecated features"
echo "  ✓ Updated or removed tests for changed functionality"
echo "  ✓ Checked for files that depend on your changes"
echo "  ✓ Ensured no duplicate functionality exists"
echo ""
echo "If any deprecated code remains, you must remove it NOW before proceeding."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"