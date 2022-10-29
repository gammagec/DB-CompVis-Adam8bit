export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

SUBJECT=beavis
DATA_ROOT=/mnt/d/dev/training_data/beavis2
CLASS_WORD=teenager
REG_DATA_ROOT=/mnt/d/dev/training_data/teenager
ACTUAL_RESUME=/mnt/c/dev/stable-diffusion-webui/models/Stable-diffusion/v1-5-pruned-pruned.ckpt
CONFIG=/mnt/c/dev/DB-CompVis-Adam8bit/configs/v1-finetune_unfrozen.yaml
MAX_STEPS=10000

set -x
python main.py -t --seed 30 --name $SUBJECT --no-test --data_root $DATA_ROOT --reg_data_root $REG_DATA_ROOT \
	--class_word $CLASS_WORD --actual_resume $ACTUAL_RESUME --base $CONFIG --subject $SUBJECT --max_steps $MAX_STEPS
