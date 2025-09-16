set -e

# nohup pulsar2 build --input hift_50_first.onnx --config config_hift_50_first.json --output_dir build-output-hift_50_first-0905 --output_name hift_50_first.axmodel --target_hardware AX650 --compiler.check  3 --compiler.check_cosine_simularity 0.98 > build_hift_50_first.log & #--debug.disable_auto_set_extremum_op_fp32 true

# nohup pulsar2 build --input hift_58.onnx --config config_hift_58.json --output_dir build-output-hift_58-0905 --output_name hift_58.axmodel --target_hardware AX650 --compiler.check  3 --compiler.check_cosine_simularity 0.98  > build_hift_58.log &


nohup pulsar2 build --input hift_p2_50_first.onnx --config config_hift_p2_50_first.json --output_dir build-output-hift_p2_50_first-0916 --output_name hift_p2_50_first.axmodel --target_hardware AX650 --compiler.check  3 --compiler.check_cosine_simularity 0.98 > build_hift_50_first.log & #--debug.disable_auto_set_extremum_op_fp32 true

nohup pulsar2 build --input hift_p2_58.onnx --config config_hift_p2_58.json --output_dir build-output-hift_p2_58-0916 --output_name hift_p2_58.axmodel --target_hardware AX650 --compiler.check  3 --compiler.check_cosine_simularity 0.98  > build_hift_58.log &
