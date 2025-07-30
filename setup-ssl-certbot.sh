#!/bin/bash
# Setup SSL certificate using Certbot for app.lakefrontdigital.io

DOMAIN="app.lakefrontdigital.io"
EMAIL="admin@lakefrontdigital.io"  # Change this to your email

echo "Setting up Let's Encrypt SSL certificate for $DOMAIN..."

# Install certbot if not already installed
if ! command -v certbot &> /dev/null; then
    echo "Installing Certbot..."
    sudo apt-get update
    sudo apt-get install -y certbot python3-certbot-nginx
fi

# First, copy the nginx config to sites-available and enable it
echo "Setting up nginx configuration..."
sudo cp /home/azureuser/4hosts/app.lakefrontdigital.io.conf /etc/nginx/sites-available/
sudo ln -sf /etc/nginx/sites-available/app.lakefrontdigital.io.conf /etc/nginx/sites-enabled/

# Create a temporary config without SSL for initial cert generation
sudo tee /etc/nginx/sites-available/app.lakefrontdigital.io-temp.conf > /dev/null << 'EOF'
server {
    listen 80;
    listen [::]:80;
    server_name app.lakefrontdigital.io;

    # Allow certbot challenge
    location /.well-known/acme-challenge/ {
        root /var/www/html;
    }

    location / {
        return 301 https://$server_name$request_uri;
    }
}
EOF

# Temporarily use the non-SSL config
sudo ln -sf /etc/nginx/sites-available/app.lakefrontdigital.io-temp.conf /etc/nginx/sites-enabled/app.lakefrontdigital.io.conf

# Test nginx config
sudo nginx -t

# Reload nginx
sudo nginx -s reload

# Obtain certificate
echo "Obtaining SSL certificate from Let's Encrypt..."
sudo certbot certonly --nginx \
    --non-interactive \
    --agree-tos \
    --email "$EMAIL" \
    -d "$DOMAIN" \
    --redirect \
    --keep-until-expiring

# Update the nginx config to use Let's Encrypt certificates
sudo sed -i "s|ssl_certificate .*|ssl_certificate /etc/letsencrypt/live/$DOMAIN/fullchain.pem;|" /etc/nginx/sites-available/app.lakefrontdigital.io.conf
sudo sed -i "s|ssl_certificate_key .*|ssl_certificate_key /etc/letsencrypt/live/$DOMAIN/privkey.pem;|" /etc/nginx/sites-available/app.lakefrontdigital.io.conf

# Add SSL stapling configuration
sudo sed -i '/ssl_certificate_key/a \    ssl_stapling on;\n    ssl_stapling_verify on;\n    ssl_trusted_certificate /etc/letsencrypt/live/'"$DOMAIN"'/chain.pem;' /etc/nginx/sites-available/app.lakefrontdigital.io.conf

# Switch back to the full config
sudo ln -sf /etc/nginx/sites-available/app.lakefrontdigital.io.conf /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-available/app.lakefrontdigital.io-temp.conf

# Test nginx config again
sudo nginx -t

# Reload nginx with new config
sudo nginx -s reload

# Setup auto-renewal
echo "Setting up automatic certificate renewal..."
sudo systemctl enable certbot.timer
sudo systemctl start certbot.timer

echo "SSL certificate setup complete!"
echo ""
echo "Certificate location: /etc/letsencrypt/live/$DOMAIN/"
echo "Auto-renewal is configured via systemd timer"
echo ""
echo "Test renewal with: sudo certbot renew --dry-run"