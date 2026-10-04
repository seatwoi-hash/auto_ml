#!/bin/sh
set -eu

NEXTCLOUD_DB_PASSWORD=pre-commit-db-password \
NEXTCLOUD_ADMIN_USER=pre-commit-admin \
NEXTCLOUD_ADMIN_PASSWORD=pre-commit-admin-password \
API_KEY=pre-commit-api-key \
docker compose config --quiet
