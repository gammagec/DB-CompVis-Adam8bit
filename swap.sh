export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

#CKPT_LOC=/mnt/c/dev/stable-diffusion-webui/models/Stable-diffusion/v1-5-pruned-pruned.ckpt
CKPT_LOC=/mnt/c/dev/stable-diffusion-webui/models/Stable-diffusion/sd-v1-4.ckpt
CONFIG=/mnt/c/dev/DB-CompVis-Adam8bit/configs/v1-inference.yaml
OUT_DIR=/mnt/d/dev/swap/swap_out
VIDEO_IN=/mnt/d/dev/swap/godfather.mp4
FPS=23.98
WRITE_PICS=false
SHOW_PREVIEW=true
FRAME_SKIP=1 # between captures
SKIP_FRAMES=0  # starting point
PROMPT="arnold schwarzenegger"
STEPS=50
SCALE=12
STRENGTH=0.35

OPTS=""
if [ "$WRITE_PICS" = true ] ; then
    OPTS="$OPTS --write_pics"
fi
if [ "$SHOW_PREVIEW" = true ] ; then
    OPTS="$OPTS --show_preview"
fi

python sd_swap.py --config $CONFIG --skip_frames $SKIP_FRAMES --ckpt_loc $CKPT_LOC --out_dir $OUT_DIR --video_in $VIDEO_IN --fps $FPS --frame_skip $FRAME_SKIP --prompt "$PROMPT" --steps $STEPS --scale $SCALE --strength $STRENGTH $OPTS