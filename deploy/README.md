# Server deployment preparation

This is a shared demonstration deployment, not an isolated multi-user service. The live recognizer is process-global: visitors must take turns. Uploaded examples and corrections share one library. Do not publish it as an unrestricted production service.

The image builds the frontend and runs the Python backend. It includes only the three `.task` model files; `.dockerignore` excludes recordings, local database, secrets and experimental checkpoints. A new installation starts with an empty demonstration library. Training with PyTorch is not installed by this image.

## Requirements

A Linux server with Docker Compose, a domain pointing to it, and inbound ports 80/443. HTTPS is required for browser camera access from other computers. CPU and RAM capacity must be measured with real recordings before scaling.

Set SITE_DOMAIN, SITE_USER and SITE_PASSWORD_HASH in the server environment. Generate a password hash with `docker run --rm -it caddy:2 caddy hash-password` (interactive input). Keep the hash and credentials out of version control. Dollar signs in a Compose env file must be single-quoted.

Run `docker compose up -d --build` from the project root. Verify `docker compose ps` and `/api/v1/health` through the authenticated HTTPS domain. Verify camera permission, live recognition, uploads, persistence after restart, and visitor isolation before inviting users.

`app_data` retains uploads and the database. Back it up before updates; do not run `docker compose down -v` unless deleting stored data is intended.

## Verification status

Deployment files are prepared. A successful remote build, HTTPS endpoint and camera test have not yet been verified. No server is provisioned by these files.
