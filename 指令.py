



python train.py --dataroot /home/yang/Desktop/Model_Nov12th.3/Model/Dataset --name unet_ssm_deepliif --seghead unet_ssm --display-server http://localhost --display-id 1 --gpu-ids 0 --display-env main

python test.py --dataroot /home/yang/Desktop/Model_Nov12th.3/Model/Dataset --name swt_bc --gpu_ids 0


visdom -port 8097
http://localhost:8097

sudo fuser -k 8097/tcp && visdom -port 8097
