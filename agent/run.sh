#!/bin/sh
set -eu

if [ ! -x /opt/autoresearch-venv/bin/python ]; then
    echo "Shared Python is missing. Follow the root README.md Setup section to install it." >&2
    exit 1
fi

if [ "${1:-}" = "population" ]; then
    shift
fi

cd "$(dirname -- "$0")/.."
exec /opt/autoresearch-venv/bin/python -m agent.population "$@"
