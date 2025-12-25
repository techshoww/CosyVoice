export CUDA_VISIBLE_DEVICES=""

nohup python export_hift.py False 50 > export_hift_50.log &
nohup python export_hift.py False 100 > export_hift_100.log &
nohup python export_hift.py False 150 > export_hift_150.log &
nohup python export_hift.py True 100 > export_hift_100_final.log &