source "/home/cm-pc/.venvs/parc2026-PARC2026_pre/bin/activate"
export PYTHONPATH="/mnt/robot_ssd/PARC2026_pre/LIBERO-plus:/mnt/robot_ssd/PARC2026_pre:/mnt/robot_ssd/PARC2026_pre/compe"
export LIBERO_ROOT="/mnt/robot_ssd/PARC2026_pre/LIBERO-plus"
# wand (LIBERO) が ImageMagick を参照する。conda-forge の lib を使う
if [ -d "$HOME/miniconda3/envs/parc2026/lib" ]; then
  export LD_LIBRARY_PATH="$HOME/miniconda3/envs/parc2026/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
