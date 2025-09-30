# Multi-stage Dockerfile for Four Hosts Frontend
# Using alternative registry to avoid Docker Hub authentication issues
FROM public.ecr.aws/docker/library/node:22-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Build the application
RUN npm run build

# --- Production Stage ---
FROM public.ecr.aws/docker/library/nginx:alpine

# Install runtime dependencies
RUN apk add --no-cache \
    curl \
    tini

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf
COPY default.conf /etc/nginx/conf.d/default.conf

# Copy built application from builder (Vite builds to dist)
COPY --from=builder /app/dist /usr/share/nginx/html

# Set proper permissions for nginx user (worker processes run as nginx, not root)
RUN chown -R nginx:nginx /usr/share/nginx/html && \
    chmod -R 755 /usr/share/nginx/html

# Create nginx cache directories with proper permissions
RUN mkdir -p /var/cache/nginx/client_temp \
             /var/cache/nginx/proxy_temp \
             /var/cache/nginx/fastcgi_temp \
             /var/cache/nginx/uwsgi_temp \
             /var/cache/nginx/scgi_temp && \
    chown -R nginx:nginx /var/cache/nginx && \
    chmod -R 755 /var/cache/nginx

# Note: nginx master process runs as root (required to bind to ports 80/443)
# Worker processes run as 'nginx' user as configured in nginx.conf
# This is standard practice for nginx containers

# Expose ports
EXPOSE 80 443

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost/health || exit 1

# Use tini for proper signal handling
ENTRYPOINT ["/sbin/tini", "--"]

# Run nginx
CMD ["nginx", "-g", "daemon off;"]
