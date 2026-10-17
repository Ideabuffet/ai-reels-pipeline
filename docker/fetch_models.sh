#!/bin/bash
# Downloads the models from models.txt into $MODELS_ROOT with aria2. Files that are already complete
# (right size, no .aria2 control file) are skipped, so on a Network Volume this finishes instantly.
# State for the orchestrator: /tmp/pipeline/fetch.state = running | done | failed: <files>
set -u
LIST=${1:-/opt/pipeline/models.txt}
ROOT=${MODELS_ROOT:-/models}
P=/tmp/pipeline
mkdir -p $P
echo running > $P/fetch.state
date +%s > $P/fetch.start

complete() {  # path size
  [ -f "$1" ] && [ ! -f "$1.aria2" ] && [ "$(stat -c %s "$1")" = "$2" ]
}

input=$P/aria.txt
: > $input
grep -v '^#' "$LIST" | while read -r url rel size; do
  [ -n "$url" ] || continue
  f=$ROOT/$rel
  complete "$f" "$size" && continue
  printf '%s\n  dir=%s\n  out=%s\n' "$url" "$(dirname "$f")" "$(basename "$f")" >> $input
done

hdr=()
[ -n "${HF_TOKEN:-}" ] && hdr=(--header="Authorization: Bearer $HF_TOKEN")
for attempt in 1 2 3; do
  [ -s $input ] || break
  echo "aria2 attempt $attempt: $(grep -c '^http' $input) files"
  aria2c -x16 -s16 -j6 -c --file-allocation=none --auto-file-renaming=false --allow-overwrite=true \
    --max-tries=5 --retry-wait=5 --console-log-level=warn --summary-interval=60 "${hdr[@]}" -i $input && break
  sleep 10
done

bad=""
while read -r url rel size; do
  [ -n "$url" ] || continue
  complete "$ROOT/$rel" "$size" || bad="$bad $(basename "$rel")"
done < <(grep -v '^#' "$LIST")
date +%s > $P/fetch.end
if [ -z "$bad" ]; then
  echo done > $P/fetch.state
  echo "models ready in $(( $(cat $P/fetch.end) - $(cat $P/fetch.start) )) s"
else
  echo "failed:$bad" > $P/fetch.state
  echo "FAILED:$bad"
fi
