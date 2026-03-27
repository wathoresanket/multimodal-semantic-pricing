#!/usr/bin/env bash
# =============================================================================
#  run.sh  —  Launch any pipeline step in the background via nohup
#
#  Usage:
#    bash run.sh <step>         # run one step
#    bash run.sh all            # run all steps sequentially (each waits for prev)
#    bash run.sh status         # check which steps are running / done
#    bash run.sh logs <step>    # tail live logs for a step
#    bash run.sh kill <step>    # kill a running step
#
#  Steps:  features | text | images | train | predict
#
#  Examples:
#    bash run.sh images         # embed images in background, close terminal safely
#    bash run.sh logs images    # watch image download progress live
#    bash run.sh status         # see what's running
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
PID_DIR="$SCRIPT_DIR/logs/pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

PYTHON="${PYTHON:-python3}"   # override with: PYTHON=python bash run.sh ...

# ── Map step name → script ───────────────────────────────────────────────────
step_to_script() {
    case "$1" in
        features) echo "steps/01_extract_features.py" ;;
        text)     echo "steps/02_text_embeddings.py"  ;;
        images)   echo "steps/03_image_embeddings.py" ;;
        train)    echo "train.py"                     ;;
        predict)  echo "predict.py"                   ;;
        *)        echo ""; return 1                   ;;
    esac
}

step_log() {
    echo "$LOG_DIR/${1}.out"
}

step_pid_file() {
    echo "$PID_DIR/${1}.pid"
}

# ── Launch a step ────────────────────────────────────────────────────────────
launch() {
    local step="$1"
    local script
    script="$(step_to_script "$step")" || { echo "Unknown step: $step"; exit 1; }

    local log_file pid_file
    log_file="$(step_log "$step")"
    pid_file="$(step_pid_file "$step")"

    if [[ -f "$pid_file" ]]; then
        local old_pid
        old_pid=$(<"$pid_file")
        if kill -0 "$old_pid" 2>/dev/null; then
            echo "⚠  Step '$step' is ALREADY RUNNING (PID=$old_pid)."
            echo "   Use:  bash run.sh kill $step"
            exit 1
        fi
    fi

    echo "▶  Launching step: $step"
    echo "   Script : $SCRIPT_DIR/$script"
    echo "   Log    : $log_file"

    cd "$SCRIPT_DIR"
    nohup "$PYTHON" -u "$script" >> "$log_file" 2>&1 &
    local pid=$!
    echo "$pid" > "$pid_file"

    echo "   PID    : $pid"
    echo ""
    echo "   Watch live output:"
    echo "     tail -f $log_file"
    echo ""
    echo "   You can now safely close this terminal."
}

# ── Status ───────────────────────────────────────────────────────────────────
show_status() {
    printf "%-12s  %-8s  %-10s  %s\n" "STEP" "PID" "STATUS" "LOG"
    printf "%-12s  %-8s  %-10s  %s\n" "----" "---" "------" "---"
    for step in features text images train predict; do
        local pid_file log_file status pid_str
        pid_file="$(step_pid_file "$step")"
        log_file="$(step_log "$step")"
        if [[ -f "$pid_file" ]]; then
            pid_str=$(<"$pid_file")
            if kill -0 "$pid_str" 2>/dev/null; then
                status="RUNNING"
            else
                status="DONE/DEAD"
            fi
        else
            pid_str="-"
            status="NOT_STARTED"
        fi
        printf "%-12s  %-8s  %-10s  %s\n" "$step" "$pid_str" "$status" "${log_file##*/}"
    done
}

# ── Tail logs ────────────────────────────────────────────────────────────────
tail_logs() {
    local step="$1"
    local log_file
    log_file="$(step_log "$step")"
    if [[ ! -f "$log_file" ]]; then
        echo "No log file found: $log_file"
        exit 1
    fi
    echo "Tailing $log_file  (Ctrl+C to stop)"
    tail -f "$log_file"
}

# ── Kill a step ──────────────────────────────────────────────────────────────
kill_step() {
    local step="$1"
    local pid_file
    pid_file="$(step_pid_file "$step")"
    if [[ ! -f "$pid_file" ]]; then
        echo "No PID file found for step '$step' — maybe it was never started."
        exit 1
    fi
    local pid=$(<"$pid_file")
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid"
        echo "Killed step '$step' (PID=$pid)"
        rm -f "$pid_file"
    else
        echo "Step '$step' (PID=$pid) is not running."
    fi
}

# ── Run all steps sequentially ───────────────────────────────────────────────
run_all() {
    echo "Running all steps sequentially (this script will block until done)."
    for step in features text images train predict; do
        local script log_file
        script="$(step_to_script "$step")"
        log_file="$(step_log "$step")"
        echo ""
        echo "═══════════════════════════════════════"
        echo "  Step: $step"
        echo "═══════════════════════════════════════"
        cd "$SCRIPT_DIR"
        "$PYTHON" -u "$script" 2>&1 | tee -a "$log_file"
    done
    echo ""
    echo "✓ All steps complete."
}

# ── Dispatch ─────────────────────────────────────────────────────────────────
CMD="${1:-}"
case "$CMD" in
    features|text|images|train|predict)
        launch "$CMD"
        ;;
    all)
        run_all
        ;;
    status)
        show_status
        ;;
    logs)
        tail_logs "${2:-}"
        ;;
    kill)
        kill_step "${2:-}"
        ;;
    *)
        echo "Usage: bash run.sh <features|text|images|train|predict|all|status>"
        echo "       bash run.sh logs <step>"
        echo "       bash run.sh kill <step>"
        exit 1
        ;;
esac
