export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

NAME=ligam
SUBJECT=ligam
DATA_ROOT=/mnt/d/dev/training_data/lily_face
REG_DATA_ROOT=/mnt/d/dev/training_data/woman
CLASS_WORD=woman
#ACTUAL_RESUME=/mnt/c/dev/stable-diffusion-webui/models/Stable-diffusion/sd-v1-4.ckpt
ACTUAL_RESUME=/mnt/c/dev/DB-CompVis-Adam8bit/logs/lily_face2022-10-19T20-12-36_ligam/checkpoints/last.ckpt
CONFIG=/mnt/c/dev/DB-CompVis-Adam8bit/configs/v1-finetune_unfrozen.yaml
MAX_STEPS=8000

set -x
python main.py -t --seed 24 --name $NAME --no-test --data_root $DATA_ROOT --reg_data_root $REG_DATA_ROOT \
	--class_word $CLASS_WORD --actual_resume $ACTUAL_RESUME --base $CONFIG --subject $SUBJECT --max_steps $MAX_STEPS
