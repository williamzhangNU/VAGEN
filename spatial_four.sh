#!/usr/bin/env bash
if [ $# -ne 5 ]; then echo "Usage: $0 seed_range model_name output_dir input_dir parallel(y|n)"; exit 1; fi
# Default values
SEED="${1:-0-99}"
MODEL="${2:-gpt-5}"
output_dir="${3:-results}"
input_dir="${4:-room_data}"
PAR="$(echo "${5:-y}" | tr A-Z a-z)"
# Example usage: bash spatial_four.sh [seed_range] [model_name] [output_dir] [input_dir] [parallel (y|n)]
cd "$(dirname "$0")" || exit 1
mkdir -p logs
AT="ActiveDir,ActiveFwdFov,ActiveBwdNav,ActiveE2A,ActiveRot,ActiveRotDual,ActiveFwdLoc,ActiveBwdLoc"
PT="PassiveDir,PassiveFwdFov,PassiveBwdNav,PassiveE2A,PassiveRot,PassiveRotDual,PassiveFwdLoc,PassiveBwdLoc"

launch() {
  TYPE="$1"; MODE="$2"; AGT="$3"
  [ "$TYPE" = "active" ] && TASKS="$AT" || TASKS="$PT"
  EXTRA=""; if [ "$TYPE" = "passive" ]; then if [ -n "$AGT" ]; then EXTRA="--proxy-agent $AGT"; else [ "$MODE" = "text" ] && EXTRA="--proxy-agent strategist" || EXTRA="--proxy-agent scout"; fi; fi
  if [ -n "$AGT" ]; then SESS="${MODEL}_${TYPE}_${MODE}_${AGT}_${SEED}"; LOG="logs/${MODEL}-${TYPE}-${MODE}-${AGT}-${SEED}.log"; else SESS="${MODEL}_${TYPE}_${MODE}_${SEED}"; LOG="logs/${MODEL}-${TYPE}-${MODE}-${SEED}.log"; fi
  CMD="conda activate vagen; python scripts/SpatialGym/spatial_run.py --tasks $TASKS --model_name $MODEL --seed-range $SEED --render-mode $MODE --inference-only $EXTRA --output-root $output_dir --data-dir $input_dir 2>&1 | tee $LOG"
  echo "[run CMD]: $CMD"; echo "[tmux session]: $SESS"; echo "run tmux command to attach to session: tmux a -t $SESS"
  if tmux has-session -t "$SESS" 2>/dev/null; then return; fi
  tmux new-session -d -s "$SESS"
  tmux send-keys -t "$SESS" "$CMD" C-m
}

wait_session() { while tmux has-session -t "$1" 2>/dev/null; do sleep 5; done; }

if [ "$PAR" = "y" ] || [ "$PAR" = "yes" ] || [ "$PAR" = "1" ]; then # parallel run
  launch active text;
  sleep 5

  # if [ "$MODEL" != "internvl3_5" ] && [ "$MODEL" != "gpt-oss-120b" ] && [ "$MODEL" != "gpt-oss-20b" ]; then
  #   launch active vision
  #   sleep 5
  # fi

  # launch passive text strategist
  # sleep 5

  # # if [ "$MODEL" = "gpt-5" ]; then
  # #   launch passive text oracle
  # #   sleep 5
  # #   launch passive text scout
  # #   sleep 5
  # # fi

  # if [ "$MODEL" != "gpt-oss-120b" ] && [ "$MODEL" != "gpt-oss-20b" ]; then
  #   launch passive vision
  # fi
  echo "launched runs in tmux (logs under VAGEN/logs)"

else # sequential run
  launch active text; wait_session "${MODEL}_active_text_${SEED}"

  if [ "$MODEL" != "internvl3_5" ] && [ "$MODEL" != "gpt-oss-120b" ] && [ "$MODEL" != "gpt-oss-20b" ]; then
    launch active vision; wait_session "${MODEL}_active_vision_${SEED}"
  fi

  launch passive text; wait_session "${MODEL}_passive_text_${SEED}"

  if [ "$MODEL" = "gpt-5" ]; then
    # launch passive text oracle; wait_session "${MODEL}_passive_text_oracle_${SEED}"
    launch passive text strategist; wait_session "${MODEL}_passive_text_strategist_${SEED}"
    launch passive text scout; wait_session "${MODEL}_passive_text_scout_${SEED}"
  fi

  if [ "$MODEL" != "gpt-oss-120b" ] && [ "$MODEL" != "gpt-oss-20b" ]; then
    launch passive vision; wait_session "${MODEL}_passive_vision_${SEED}"
  fi
fi
