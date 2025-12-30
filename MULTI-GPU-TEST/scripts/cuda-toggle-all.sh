#!/usr/bin/env bash
set -euo pipefail

TARGET_STATE="${1:-}"
if [[ "${TARGET_STATE}" != "running" && "${TARGET_STATE}" != "checkpointed" ]]; then
  echo "usage: cuda-toggle-all <running|checkpointed>" >&2
  exit 64
fi

VERIFY_ATTEMPTS="${CUDA_TOGGLE_VERIFY_ATTEMPTS:-50}"
VERIFY_SLEEP_S="${CUDA_TOGGLE_VERIFY_SLEEP_S:-0.2}"

main_pid="$(pgrep -f "vllm.entrypoints.openai.api_server" | head -n1 || true)"
if [[ -z "${main_pid}" ]]; then
  echo "[cuda-toggle-all] ERROR: vLLM api_server pid not found" >&2
  exit 1
fi

echo "[cuda-toggle-all] main pid: ${main_pid}" >&2

gather_all_pids() {
  python3 - <<PY
import subprocess
main=int("${main_pid}")
out=subprocess.check_output(["ps","-e","-o","pid=,ppid="], text=True).strip().splitlines()
ppid_map={}
for line in out:
    pid_s, ppid_s = line.split()
    ppid_map.setdefault(int(ppid_s), []).append(int(pid_s))
seen=set()
q=[main]
while q:
    p=q.pop(0)
    if p in seen:
        continue
    seen.add(p)
    q.extend(ppid_map.get(p, []))
print(" ".join(str(x) for x in sorted(seen)))
PY
}

cuda_state_of_pid() {
  local pid="$1"
  # normalize output to lowercase single token
  cuda-checkpoint --get-state --pid "${pid}" 2>/dev/null \
    | tr -d "\r" \
    | awk '{print tolower($1)}' \
    | head -n1 \
    || true
}

# Identify CUDA-controllable PIDs from the vLLM process tree.
# We treat both "running" and "checkpointed" as eligible.
get_cuda_pids() {
  local all_pids
  all_pids="$(gather_all_pids)"
  local pid st
  for pid in ${all_pids}; do
    st="$(cuda_state_of_pid "${pid}")"
    if [[ "${st}" == "running" || "${st}" == "checkpointed" ]]; then
      echo "${pid}"
    fi
  done
}

# Loop: toggle any pid not in target state, until all are in target.
for attempt in $(seq 1 "${VERIFY_ATTEMPTS}"); do
  mapfile -t cuda_pids < <(get_cuda_pids)

  if [[ "${#cuda_pids[@]}" -eq 0 ]]; then
    echo "[cuda-toggle-all] ERROR: no CUDA-controllable PIDs found" >&2
    exit 2
  fi

  all_ok=1
  for pid in "${cuda_pids[@]}"; do
    st="$(cuda_state_of_pid "${pid}")"
    if [[ "${st}" != "${TARGET_STATE}" ]]; then
      all_ok=0
      if ! cuda-checkpoint --toggle --pid "${pid}" >/dev/null 2>&1; then
        echo "[cuda-toggle-all] WARN: toggle failed for pid=${pid} (state=${st})" >&2
      fi
    fi
  done

  if [[ "${all_ok}" -eq 1 ]]; then
    echo "[cuda-toggle-all] OK: all CUDA PIDs are '${TARGET_STATE}'" >&2
    exit 0
  fi

  sleep "${VERIFY_SLEEP_S}"
done

echo "[cuda-toggle-all] ERROR: did not reach target state '${TARGET_STATE}' after ${VERIFY_ATTEMPTS} attempts" >&2

echo "[cuda-toggle-all] final states:" >&2
for pid in $(get_cuda_pids); do
  echo "  pid=${pid} state=$(cuda_state_of_pid "${pid}")" >&2
done

exit 3
