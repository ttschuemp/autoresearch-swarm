#!/bin/sh
set -eu

if [ ! -x /opt/autoresearch-venv/bin/python ]; then
    echo "Shared Python is missing. Follow the root README.md Setup section to install it." >&2
    exit 1
fi

exec /opt/autoresearch-venv/bin/python "$(dirname -- "$0")/main.py" "$@"
