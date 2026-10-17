#!/bin/bash
# Dead man's switch. The orchestrator pings /pipeline/status every few seconds; if the pings stop
# (Mac asleep, crashed, network lost) for PIPELINE_HEARTBEAT_MIN, or the pod has lived longer than
# PIPELINE_MAX_MIN, the pod stops itself so it cannot keep billing unattended.
# Uses the pod-scoped RUNPOD_API_KEY / RUNPOD_POD_ID that RunPod injects into every pod.
[ "${PIPELINE_WATCHDOG:-1}" = 0 ] && { echo "watchdog disabled"; exit 0; }
HB=/tmp/pipeline/heartbeat
IDLE=$(( ${PIPELINE_HEARTBEAT_MIN:-20} * 60 ))
MAX=$(( ${PIPELINE_MAX_MIN:-240} * 60 ))
mkdir -p /tmp/pipeline
touch $HB
start=$(date +%s)

self_stop() {
  if [ -z "${RUNPOD_POD_ID:-}" ] || [ -z "${RUNPOD_API_KEY:-}" ]; then
    echo "no RUNPOD_POD_ID / RUNPOD_API_KEY in env, cannot stop the pod"; return 1
  fi
  curl -sf -X POST "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop" \
       -H "Authorization: Bearer $RUNPOD_API_KEY" -o /dev/null && return 0
  curl -sf https://api.runpod.io/graphql -H "Content-Type: application/json" \
       -H "Authorization: Bearer $RUNPOD_API_KEY" \
       -d "{\"query\":\"mutation { podStop(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) { id } }\"}" -o /dev/null
}

echo "watchdog: heartbeat timeout ${IDLE}s, max lifetime ${MAX}s"
while sleep 30; do
  now=$(date +%s)
  idle=$(( now - $(stat -c %Y $HB) ))
  if (( idle > IDLE || now - start > MAX )); then
    echo "$(date -u +%T) stopping pod: idle ${idle}s, alive $(( now - start ))s"
    self_stop && echo "stop requested" || echo "stop failed, retrying"
    sleep 90
  fi
done
