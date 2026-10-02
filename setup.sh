#!/usr/bin/env bash
# One-time setup: espeak-ng (lyrics -> phonemes), DiffSinger voicebanks (singing),
# MBROLA + voices (old engine), Python env.
set -euo pipefail
cd "$(dirname "$0")"

command -v espeak-ng >/dev/null || brew install espeak-ng

mkdir -p vendor/voices
if [ ! -x vendor/MBROLA/Bin/mbrola ]; then
  [ -d vendor/MBROLA ] || git clone --depth 1 https://github.com/numediart/MBROLA.git vendor/MBROLA
  make -C vendor/MBROLA
fi
for v in us1 us2 us3; do
  [ -f "vendor/voices/$v" ] || curl -fsSL -o "vendor/voices/$v" "https://github.com/numediart/MBROLA-voices/raw/master/data/$v/$v"
done

# DiffSinger voicebanks from the LUNAI Project (~430 MB each; non-commercial use with credit,
# see the terms of use inside each bank)
for v in Katyusha Keiro_Revenant Liam_Thorne; do
  d="vendor/voicebanks/$(echo "$v" | tr 'A-Z' 'a-z')"
  if [ ! -f "$d/configs/dsconfig.yaml" ]; then
    mkdir -p vendor/voicebanks
    curl -fL -o "$d.zip" "https://github.com/lunaiproject/lunai_singers/releases/download/170/${v}_v170.zip"
    ditto -x -k "$d.zip" "$d" && rm "$d.zip"
  fi
done

[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt
echo "Ready. Run: .venv/bin/python sing.py"
