#!/usr/bin/env bash
set -euo pipefail
DOMAIN="${1:-}"
EMAIL="${2:-}"
if [[ -z "${DOMAIN}" ]]; then
  echo "Usage: $0 <domain> [email]" >&2
  exit 1
fi
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LE_DIR="$ROOT_DIR/frontend/letsencrypt"
WEBROOT_DIR="$ROOT_DIR/frontend/certbot"
OUT_DIR="$ROOT_DIR/frontend/certs"
mkdir -p "$LE_DIR" "$WEBROOT_DIR" "$OUT_DIR"
CMD=(certbot certonly --webroot -w /var/www/certbot -d "$DOMAIN" --agree-tos --non-interactive)
if [[ -n "${EMAIL}" ]]; then
  CMD+=(--email "$EMAIL")
else
  CMD+=(--register-unsafely-without-email)
fi

docker run --rm   -v "$LE_DIR":/etc/letsencrypt   -v "$WEBROOT_DIR":/var/www/certbot   certbot/certbot:latest   "${CMD[@]}"

docker run --rm   -v "$LE_DIR":/etc/letsencrypt   -v "$OUT_DIR":/cert-out   alpine:3.20 sh -c "cp -f /etc/letsencrypt/live/$DOMAIN/fullchain.pem /cert-out/fullchain.pem && cp -f /etc/letsencrypt/live/$DOMAIN/privkey.pem /cert-out/privkey.pem"

echo "Certificates copied to $OUT_DIR. Restarting frontend..."
cd "$ROOT_DIR" && docker compose restart fourhosts-frontend || true
echo "Done."
