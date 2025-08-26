export CUDA_VISIBLE_DEVICES=0

nohup python model_convert/export_hift.py True 50 > export_hift_first_50.log &
nohup python model_convert/export_hift.py False 58 > export_hift_58.log &