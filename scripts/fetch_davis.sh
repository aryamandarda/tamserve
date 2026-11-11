#!/usr/bin/env bash
# Fetch DAVIS 2017 train+val at 480p (833 MB) into data/davis. Never committed.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/data/davis"

if [ -d "$DEST/DAVIS" ]; then
  echo "DAVIS already at $DEST/DAVIS"
  exit 0
fi
mkdir -p "$DEST"
curl -fL -o "$DEST/davis.zip" \
  https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
unzip -q "$DEST/davis.zip" -d "$DEST"
rm "$DEST/davis.zip"
echo "DAVIS ready at $DEST/DAVIS"
