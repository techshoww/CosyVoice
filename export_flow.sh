export CUDA_VISIBLE_DEVICES=""
nohup python -u export_flow_encoder.py 28 False > export_flow_28.log &
nohup python -u export_flow_encoder.py 53 False > export_flow_53.log &
nohup python -u export_flow_encoder.py 78 False > export_flow_78.log &
nohup python -u export_flow_encoder.py 50 True > export_flow_50_final.log &

nohup python -u export_flow_estimator.py 200 > export_est_200.log &
nohup python -u export_flow_estimator.py 250 > export_est_250.log &
nohup python -u export_flow_estimator.py 300 > export_est_300.log &