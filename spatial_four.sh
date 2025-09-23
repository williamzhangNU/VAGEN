#!/usr/bin/env bash
if [ $# -ne 3 ]; then echo "Usage: $0 seed_range model_name parallel(y|n)"; exit 1; fi
SEED="$1"; MODEL="$2"; PAR="$(echo "$3" | tr A-Z a-z)"
cd "$(dirname "$0")" || exit 1
mkdir -p logs
AT="ActiveDir,ActivePov,ActiveBwdPov,ActiveFwdFov,ActiveBwdNav,ActiveE2A,ActiveRot,ActiveRotDual,ActiveFwdLoc,ActiveBwdLoc"
PT="PassiveDir,PassivePov,PassiveBwdPov,PassiveFwdFov,PassiveBwdNav,PassiveE2A,PassiveRot,PassiveRotDual,PassiveFwdLoc,PassiveBwdLoc"

launch() {
  TYPE="$1"; MODE="$2"
  [ "$TYPE" = "active" ] && TASKS="$AT" || TASKS="$PT"
  EXTRA=""; if [ "$TYPE" = "passive" ]; then [ "$MODE" = "text" ] && EXTRA="--proxy-agent strategist" || EXTRA="--proxy-agent scout"; fi
  SESS="${MODEL}_${TYPE}_${MODE}_${SEED}"; LOG="logs/${MODEL}-${TYPE}-${MODE}.log"
  CMD="python scripts/SpatialGym/spatial_run.py --tasks $TASKS --model_name $MODEL --seed-range $SEED --render-mode $MODE --inference-only $EXTRA 2>&1 | tee $LOG"
  echo "[run CMD]: $CMD"; echo "[tmux session]: $SESS"; echo "run tmux command to attach to session: tmux a -t $SESS"
  if tmux has-session -t "$SESS" 2>/dev/null; then return; fi
  tmux new-session -d -s "$SESS"
  tmux send-keys -t "$SESS" "$CMD" C-m
}

wait_session() { while tmux has-session -t "$1" 2>/dev/null; do sleep 5; done; }

if [ "$PAR" = "y" ] || [ "$PAR" = "yes" ] || [ "$PAR" = "1" ]; then
  launch active text
  sleep 10
  launch active vision
  sleep 10
  launch passive text
  sleep 10
  launch passive vision
  echo "launched 4 runs in tmux (logs under VAGEN/logs)"
else
  launch active text; wait_session "sg_${MODEL}_active_text_${SEED}"
  launch active vision; wait_session "sg_${MODEL}_active_vision_${SEED}"
  launch passive text; wait_session "sg_${MODEL}_passive_text_${SEED}"
  launch passive vision; wait_session "sg_${MODEL}_passive_vision_${SEED}"
fi

