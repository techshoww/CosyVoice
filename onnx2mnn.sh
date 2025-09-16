set -e 

# pip install MNN==3.2.3

mnnconvert -f ONNX --modelFile hift_p1_50_first.onnx --MNNModel hift_p1_50_first.mnn --bizCode hift_p1_50_first

mnnconvert -f ONNX --modelFile hift_p1_58.onnx --MNNModel hift_p1_58.mnn --bizCode hift_p1_58