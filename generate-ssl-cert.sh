#!/bin/bash
# Generate self-signed SSL certificate for app.lakefrontdigital.io

DOMAIN="app.lakefrontdigital.io"
SSL_DIR="/etc/nginx/ssl-certificates"

echo "Creating self-signed SSL certificate for $DOMAIN..."

# Create SSL directory if it doesn't exist
sudo mkdir -p "$SSL_DIR"

# Generate private key and certificate
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout "$SSL_DIR/$DOMAIN.key" \
    -out "$SSL_DIR/$DOMAIN.crt" \
    -subj "/C=US/ST=State/L=City/O=Organization/CN=$DOMAIN"

# Set proper permissions
sudo chmod 600 "$SSL_DIR/$DOMAIN.key"
sudo chmod 644 "$SSL_DIR/$DOMAIN.crt"

echo "SSL certificate generated successfully!"
echo "Certificate: $SSL_DIR/$DOMAIN.crt"
echo "Private Key: $SSL_DIR/$DOMAIN.key"