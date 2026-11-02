#!/bin/sh
set -eu

NEXTCLOUD_DB_PASSWORD=pre-commit-db-password \
NEXTCLOUD_USER=pre-commit-admin \
NEXTCLOUD_PASSWORD=pre-commit-admin-password \
docker compose config --quiet
