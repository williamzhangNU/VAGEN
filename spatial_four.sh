#!/usr/bin/env bash
if [ $# -ne 3 ]; then echo "Usage: $0 seed_range model_name parallel(y|n)"; exit 1; fi
SEED="$1"; MODEL="$2"; PAR="$(echo "$3" | tr A-Z a-z)"
cd "$(dirname "$0")" || exit 1
mkdir -p logs
AT="ActiveDir,ActivePov,ActiveBwdPov,ActiveFwdFov,ActiveBwdNav,ActiveE2A,ActiveRot,ActiveRotDual,ActiveFwdLoc,ActiveBwdLoc"
PT="PassiveDir,PassivePov,PassiveBwdPov,PassiveFwdFov,PassiveBwdNav,PassiveE2A,PassiveRot,PassiveRotDual,PassiveFwdLoc,PassiveBwdLoc"

launch() {
  TYPE="$1"; MODE="$2"; AGT="$3"
  [ "$TYPE" = "active" ] && TASKS="$AT" || TASKS="$PT"
  EXTRA=""; if [ "$TYPE" = "passive" ]; then if [ -n "$AGT" ]; then EXTRA="--proxy-agent $AGT"; else [ "$MODE" = "text" ] && EXTRA="--proxy-agent strategist" || EXTRA="--proxy-agent scout"; fi; fi
  if [ -n "$AGT" ]; then SESS="${MODEL}_${TYPE}_${MODE}_${AGT}_${SEED}"; LOG="logs/${MODEL}-${TYPE}-${MODE}-${AGT}-${SEED}.log"; else SESS="${MODEL}_${TYPE}_${MODE}_${SEED}"; LOG="logs/${MODEL}-${TYPE}-${MODE}-${SEED}.log"; fi
  CMD="python scripts/SpatialGym/spatial_run.py --eval-override --eval-override-tasks dir --tasks $TASKS --model_name $MODEL --seed-range $SEED --render-mode $MODE --inference-only $EXTRA 2>&1 | tee $LOG"
  echo "[run CMD]: $CMD"; echo "[tmux session]: $SESS"; echo "run tmux command to attach to session: tmux a -t $SESS"
  if tmux has-session -t "$SESS" 2>/dev/null; then return; fi
  tmux new-session -d -s "$SESS"
  tmux send-keys -t "$SESS" "$CMD" C-m
}

wait_session() { while tmux has-session -t "$1" 2>/dev/null; do sleep 5; done; }

if [ "$PAR" = "y" ] || [ "$PAR" = "yes" ] || [ "$PAR" = "1" ]; then
  launch active text
  sleep 10
  if [ "$MODEL" != "internvl3_5" ] && [ "$MODEL" != "gpt-oss-120b" ] && [ "$MODEL" != "gpt-oss-20b" ]; then
    launch active vision
    sleep 10
  fi
  launch passive text
  sleep 10
  if [ "$MODEL" = "gpt-5" ]; then
    launch passive text oracle
    sleep 10
    launch passive text scout
    sleep 10
  fi
  if [ "$MODEL" != "gpt-oss-120b" ] && [ "$MODEL" != "gpt-oss-20b" ]; then
    launch passive vision
  fi
  echo "launched runs in tmux (logs under VAGEN/logs)"
else
  launch active text; wait_session "${MODEL}_active_text_${SEED}"
  if [ "$MODEL" != "internvl3_5" ] && [ "$MODEL" != "gpt-oss-120b" ] && [ "$MODEL" != "gpt-oss-20b" ]; then
    launch active vision; wait_session "${MODEL}_active_vision_${SEED}"
  fi
  launch passive text; wait_session "${MODEL}_passive_text_${SEED}"
  if [ "$MODEL" = "gpt-5" ]; then
    launch passive text oracle; wait_session "${MODEL}_passive_text_oracle_${SEED}"
    launch passive text scout; wait_session "${MODEL}_passive_text_scout_${SEED}"
  fi
  if [ "$MODEL" != "gpt-oss-120b" ] && [ "$MODEL" != "gpt-oss-20b" ]; then
    launch passive vision; wait_session "${MODEL}_passive_vision_${SEED}"
  fi
fi
