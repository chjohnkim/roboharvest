cd /home/amiga-bak/johnkim/release_testing/roboharvest
sudo chmod -R 777 /dev/bus/usb
sudo chmod -R 666 /dev/ttyUSB0 
/home/amiga-bak/miniconda3/envs/umi/bin/python scripts/eval_real.py --robot_config=config/eval_robots_config.yaml -i ./ckpts/epoch=0085-train_loss=0.013.ckpt -o data/eval_temp