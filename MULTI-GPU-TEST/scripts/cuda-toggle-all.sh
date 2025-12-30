#!/usr/bin/env bash
set -euo pipefail

TARGET_STATE="${1:-}"
if [[ "${TARGET_STATE}" != "running" && "${TARGET_STATE}" != "checkpointed" ]]; then
  echo "usage: cuda-toggle-all <running|checkpointed>" >&2
  exit 64
fi

VERIFY_ATTEMPTS="${CUDA_TOGGLE_VERIFY_ATTEMPTS:-50}"
VERIFY_SLEEP_S="${CUDA_TOGGLE_VERIFY_SLEEP_S:-0.2}"
TOGGLE_PARALLEL="${CUDA_TOGGLE_PARALLEL:-1}"
TOGGLE_MAX_WORKERS="${CUDA_TOGGLE_MAX_WORKERS:-0}" # 0 => auto

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

toggle_one_pid() {
  local pid="$1"
  local st_before="$2"

  if ! cuda-checkpoint --toggle --pid "${pid}" >/dev/null 2>&1; then
    echo "[cuda-toggle-all] WARN: toggle failed for pid=${pid} (state=${st_before})" >&2
    return 1
  fi
  return 0
}

toggle_pids_parallel() {
  local -a pids=("$@")
  local max_workers="${TOGGLE_MAX_WORKERS}"

  if [[ "${max_workers}" -le 0 ]]; then
    # Safe default: small parallelism; enough to cover TP=2 without overloading.
    if [[ "${#pids[@]}" -le 8 ]]; then
      max_workers="${#pids[@]}"
    else
      max_workers=8
    fi
  fi

  # Launch with bounded concurrency.
  local pid st
  for pid in "${pids[@]}"; do
    st="$(cuda_state_of_pid "${pid}")"
    toggle_one_pid "${pid}" "${st}" &

    # If we're at the limit, wait for one job to finish.
    while [[ "$(jobs -rp | wc -l)" -ge "${max_workers}" ]]; do
      wait -n || true
    done
  done

  # Wait for remaining jobs.
  while [[ "$(jobs -rp | wc -l)" -gt 0 ]]; do
    wait -n || true
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
  to_toggle=()
  for pid in "${cuda_pids[@]}"; do
    st="$(cuda_state_of_pid "${pid}")"
    if [[ "${st}" != "${TARGET_STATE}" ]]; then
      all_ok=0
      to_toggle+=("${pid}")
    fi
  done

  if [[ "${all_ok}" -eq 0 && "${#to_toggle[@]}" -gt 0 ]]; then
    if [[ "${TOGGLE_PARALLEL}" == "1" || "${TOGGLE_PARALLEL,,}" == "true" || "${TOGGLE_PARALLEL,,}" == "yes" ]]; then
      toggle_pids_parallel "${to_toggle[@]}"
    else
      for pid in "${to_toggle[@]}"; do
        st="$(cuda_state_of_pid "${pid}")"
        toggle_one_pid "${pid}" "${st}" || true
      done
    fi
  fi

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
