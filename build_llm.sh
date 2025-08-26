set -e 

INPUT=pretrained_models/CosyVoice2-0.5B
OUTPUT=CosyVoice-BlankEN-Ax650-prefill_512

timestamp=$(date +%s) # 获取当前时间戳
INPUT_MID="${INPUT}_${timestamp}"

echo $INPUT_MID

cp -r $INPUT $INPUT_MID

python pt_to_safetensors.py  $INPUT $INPUT_MID


CKPT=$INPUT_MID/CosyVoice-BlankEN
pulsar2 llm_build --input_path $CKPT \
                --output_path $OUTPUT \
                --kv_cache_len 1023 \
                --hidden_state_type bf16 \
                --prefill_len 128 \
                --last_kv_cache_len 128 \
                --last_kv_cache_len 256 \
                --last_kv_cache_len 384 \
                --last_kv_cache_len 512 \
                --parallel 16 \
                --ret_postnorm \
                --chip AX650

./tools/embed_process.sh $CKPT $OUTPUT
# (151936, 896)

rm -r $INPUT_MID

pulsar2 build --input llm_decoder.onnx --config config_llm_decoder.json --output_dir build-output-llm_decoder --output_name llm_decoder.axmodel --target_hardware AX650 --compiler.check  2
cp build-output-llm_decoder/llm_decoder.axmodel $OUTPUT

cp llm.llm_embedding.* $OUTPUT
cp llm.speech_embedding.* $OUTPUT