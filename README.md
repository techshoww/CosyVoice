# CosyVoice2 模型转换  

## 模型架构
```
CosyVoice2  
├── frontend  
    ├── QwenTokenizer               # prompt text 和 input text => text token  
    ├── speech_tokenizer            # prompt audio => prompt speech token  
    ├── feat_extractor              # prompt audio => prompt feature  
    └── campplus                    # prompt audio => prompt embedding  
└── tts  
    ├── llm                         # text token, prompt text token, prompt speech token, prompt embedding => text speech token  
    │     ├── Qwen2LM   
    │     └── llm decoder           # 替代原qwen2模型中的 post layer  
    └── token2wav                   # speech token => audio  
        ├── flow                    # speech token => mel  
        │     ├── encoder         
        │     └── decoder  
        │            └── estimator   
        └── hift                    # mel => audio   
```

### 一、准备工作  

1. 下载模型  
(1) 在python中安装modelscope  
```bash
pip install modelscope
```
(2) 执行以下命令
```bash
modelscope download --model iic/CosyVoice2-0.5B --local_dir pretrained_models/CosyVoice2-0.5B
modelscope download --model pengzhendong/wetext --local_dir pengzhendong/wetext
```

2. 安装python包   
```bash
pip install -r requirements.txt
``` 

### 二、生成calibration dataset  
1. 执行推理，保存所需的tensor数据  
```bash
python test.py
```
2. 将保存的tensor数据转换为npy，并打包为tar包  
```bash
python gen_calib_data.py
```

### 三、导出 onnx  
1. 导出 flow encoder 和  flow decoder estimator  
```bash
bash export_flow.sh 
```
2. 导出 hift  
```bash
bash export_hift.sh
```
3. 导出llm decoder  
```bash
python  export_llm_decoder.py
```
4. 提取llm和flow中的embedding权重  
```bash
python extract_embedding.py
```
5. 生成token2wav初始化数据  
```
python gen_init_data.py
```

### 四、转换模型（onnx -> axmodel）  
**先切换到AX650工具链环境下** 
1. 转换 flow encoder 和 flow decoder estimator  
```bash
bash build_flow.sh
```
将转换好的axmodel模型复制到当前目录下  
2. 转换 hift  
(1). 将 hift的前半段 part1 转换为MNN模型  
前半段 SineGen2 的_f02sine 量化误差较大，使用float模型在cpu上推理  
```
bash onnx2mnn.sh
```
(2). 将 hift的后半段 part2 转换为axmodel模型  
```bash
bash build_hift.sh
```
将转换好的axmodel模型复制到当前目录下  
3. 转换 llm  
```bash
bash build_llm.sh
```
4. 将编译好的axmodel复制出来放到token2wav-axmodels文件  
```
mkdir token2wav-axmodels && \
cp build-output-estimator-200/flow_estimator_200.axmodel \
build-output-estimator-250/flow_estimator_250.axmodel \
build-output-estimator-300/flow_estimator_300.axmodel \
build-output-flow_encoder_28-0825/flow_encoder_28.axmodel \
build-output-flow_encoder_50_final-0825/flow_encoder_50_final.axmodel \
build-output-flow_encoder_53-0825/flow_encoder_53.axmodel \
build-output-flow_encoder_78-0825/flow_encoder_78.axmodel \
build-output-hift_50_first-0826/hift_50_first.axmodel \
build-output-hift_58-0826/hift_58.axmodel \
flow.input_embedding.float16.bin  \
flow.input_embedding.npy \
hift_p1_50_first.mnn \
hift_p1_58.mnn \
rand_noise_1_80_300.txt \
speech_window_2x8x480.txt \
token2wav-axmodels
```

### 五、测试axmodel  
将当前文件夹下所有文件全部复制到开发板（推荐用nfs挂载）  
1. 在开发板上安装环境  
```bash
pip3 install requirements.txt
```
2. 运行测试  
```bash
python3 test_ax.py
```