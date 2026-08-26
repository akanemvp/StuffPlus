#!/usr/bin/env bash
# Start the PitchGrader web app locally on http://localhost:5001
# Usage:  ./start.sh
cd "$(dirname "$0")"
echo "PitchGrader → http://localhost:5001  (Ctrl-C to stop)"
exec python3 app.py
