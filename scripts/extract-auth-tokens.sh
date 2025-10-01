#!/bin/bash
# Extract both access_token and CSRF token from browser session

set -e

echo "=== Token Extraction Guide ==="
echo ""
echo "📋 Copy and paste this JavaScript into your browser console while logged in:"
echo ""
cat << 'EOF'
// Run this in browser console after login
(function() {
  const accessToken = localStorage.getItem('access_token') ||
                      sessionStorage.getItem('access_token') ||
                      document.cookie.match(/access_token=([^;]+)/)?.[1];

  const csrfToken = localStorage.getItem('csrf_token') ||
                    sessionStorage.getItem('csrf_token') ||
                    document.querySelector('meta[name="csrf-token"]')?.content ||
                    document.cookie.match(/csrf_token=([^;]+)/)?.[1];

  console.log('=== Export these to your shell ===\n');
  console.log(`export AUTH_TOKEN="${accessToken || 'NOT_FOUND'}"`);
  console.log(`export X_CSRF_TOKEN="${csrfToken || 'NOT_FOUND'}"\n`);

  // Copy to clipboard if available
  if (navigator.clipboard && accessToken && csrfToken) {
    const exports = `export AUTH_TOKEN="${accessToken}"\nexport X_CSRF_TOKEN="${csrfToken}"`;
    navigator.clipboard.writeText(exports);
    console.log('✓ Copied to clipboard!');
  }

  return { accessToken, csrfToken };
})();
EOF

echo ""
echo "📝 Then paste the output here or in your terminal:"
echo "   export AUTH_TOKEN=\"...\""
echo "   export X_CSRF_TOKEN=\"...\""
echo ""
echo "Alternative: Run scripts/login-and-test.sh to login via API"
