#!/usr/bin/env bash
if [ $# -ne 4 ]; then echo "Usage: $0 seed_range text|vision active|passive model_name"; exit 1; fi
SEED="$1"; MODE="$(echo "$2" | tr A-Z a-z)"; TYPE="$(echo "$3" | tr A-Z a-z)"; MODEL="$4"
cd "$(dirname "$0")"
[ "$TYPE" = "active" ] && TASKS="ActiveDir,ActivePov,ActiveBwdPov,ActiveFwdFov,ActiveBwdNav,ActiveE2A,ActiveRot,ActiveRotDual,ActiveFwdLoc,ActiveBwdLoc" || TASKS="PassiveDir,PassivePov,PassiveBwdPov,PassiveFwdFov,PassiveBwdNav,PassiveE2A,PassiveRot,PassiveRotDual,PassiveFwdLoc,PassiveBwdLoc"
EXTRA=""
if [ "$TYPE" = "passive" ]; then
  [ "$MODE" = "text" ] && EXTRA="--proxy-agent strategist" || EXTRA="--proxy-agent scout"
fi
SESS="sg_${MODEL}_${TYPE}_${MODE}_${SEED}"
if tmux has-session -t "$SESS" 2>/dev/null; then tmux attach -t "$SESS"; exit 0; fi
mkdir -p logs
CMD="python scripts/SpatialGym/spatial_run.py --tasks $TASKS --model_name $MODEL --seed-range $SEED --render-mode $MODE --inference-only $EXTRA 2>&1 | tee logs/${MODEL}-${TYPE}-${MODE}.log"
echo "[run CMD]: $CMD"
echo "[tmux session]: $SESS"
echo "run tmux command to attach to session: tmux a -t $SESS"
tmux new-session -d -s "$SESS"
tmux send-keys -t "$SESS" "$CMD" C-m
tmux attach -t "$SESS"

