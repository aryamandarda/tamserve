#!/usr/bin/env bash
# Fetch TrackEval (the standard HOTA / IDF1 scorer) at a pinned commit into
# third_party/TrackEval, patched for current numpy. Only the benchmarks need it.
set -euo pipefail

TRACKEVAL_REF=12c8791b303e0a0b50f753af204249e622d0281a

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party/TrackEval"

if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/JonathonLuiten/TrackEval "$DEST"
fi
git -C "$DEST" checkout -q "$TRACKEVAL_REF"
git -C "$DEST" checkout -q -- trackeval

# numpy 1.24 removed the np.float / np.int / np.bool aliases TrackEval still uses;
# they always meant the builtins.
# (perl, not sed: BSD sed has no \b word boundary.)
find "$DEST/trackeval" -name '*.py' -exec perl -pi -e \
  's/\bnp\.float\b/float/g; s/\bnp\.int\b/int/g; s/\bnp\.bool\b/bool/g' {} +
! grep -rqE 'np\.(float|int|bool)\b' "$DEST/trackeval" \
  || { echo "numpy alias patch did not apply" >&2; exit 1; }

echo "TrackEval ready at $DEST ($TRACKEVAL_REF)"
