export CUDA_VISIBLE_DEVICES=""

nohup python export_hift.py True 50 > export_hift_first_50.log &
nohup python export_hift.py False 58 > export_hift_58.log &