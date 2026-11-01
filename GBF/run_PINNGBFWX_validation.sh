#!/bin/bash
set -euo pipefail

# Run from the project directory containing this wrapper.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Change this if you need a specific Conda/Python executable.
PYTHON_BIN="${PYTHON_BIN:-/home/2665592k/.conda/envs/pinns-gpu/bin/python}"

echo "Python: $PYTHON_BIN"

"$PYTHON_BIN" - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is unavailable; refusing to run the PINN on CPU.")

print("GPU:", torch.cuda.get_device_name(0))
PY

NUMERICAL_SCRIPT="NumericalGBF.py"
PINN_SCRIPT="PINNGBFWX.py"

if [[ $# -ne 14 ]]; then
    echo "Usage: $0 MASS MODE X_EXTRACT OMEGA_START OMEGA_FINAL QUERY_STEPS ADAM_STEPS LBFGS_STEPS CURRICULUM_START LOWFREQ_WEIGHT LOWFREQ_MAX_M_OMEGA LOWFREQ_POINTS RES_NORM" >&2
    exit 2
fi

MASS="$1"
MODE="$2"
X_EXTRACT="$3"
OMEGA_START="$4"
OMEGA_FINAL="$5"
QUERY_STEPS="$6"
ADAM_STEPS="$7"
LBFGS_STEPS="$8"
CURRICULUM_START="$9"
LOWFREQ_WEIGHT="${10}"
LOWFREQ_MAX_M_OMEGA="${11}"
LOWFREQ_POINTS="${12}"
ENVELOPE_M_OMEGA_B="${13}"
RES_NORM="${14}"

echo "============================================================"
echo "Starting NumericalGBF.py on dense query frequency grid"
echo "query_steps=${QUERY_STEPS}"
echo "============================================================"

"$PYTHON_BIN" "$NUMERICAL_SCRIPT" \
    --mass "$MASS" \
    --mode "$MODE" \
    --x_extract "$X_EXTRACT" \
    --omega_start "$OMEGA_START" \
    --omega_final "$OMEGA_FINAL" \
    --num_steps "$QUERY_STEPS"

echo "============================================================"
echo "Numerical solver finished successfully."
echo "Starting PINNGBFWX.py"
echo "============================================================"

PINN_ARGS=(
    --mass "$MASS"
    --mode "$MODE"
    --x_extract "$X_EXTRACT"
    --omega_start "$OMEGA_START"
    --omega_final "$OMEGA_FINAL"
    --query_steps "$QUERY_STEPS"
    --adam_steps "$ADAM_STEPS"
    --lbfgs_steps "$LBFGS_STEPS"
    --curriculum_start "$CURRICULUM_START"
    --lowfreq_weight "$LOWFREQ_WEIGHT"
    --lowfreq_max_M_omega "$LOWFREQ_MAX_M_OMEGA"
    --lowfreq_points "$LOWFREQ_POINTS"
    --envelope_M_omega_b "$ENVELOPE_M_OMEGA_B"
    --res_norm "$RES_NORM"
)


"$PYTHON_BIN" "$PINN_SCRIPT" "${PINN_ARGS[@]}"

echo "============================================================"
echo "NumericalGBF + PINNGBFWX validation run complete."
echo "============================================================"
