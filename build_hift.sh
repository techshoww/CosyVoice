set -e

pulsar2 build --input hift_50_first.onnx --config config_u16.json --output_dir build-output-hift_50_first --output_name hift_50_first.axmodel --target_hardware AX650 --compiler.check  3 --compiler.check_cosine_simularity 0.98 #--debug.disable_auto_set_extremum_op_fp32 true

pulsar2 build --input hift_58.onnx --config config_u16.json --output_dir build-output-hift_58 --output_name hift_58.axmodel --target_hardware AX650 --compiler.check  3 --compiler.check_cosine_simularity 0.98 #--debug.disable_auto_set_extremum_op_fp32 true
