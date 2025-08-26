# CosyVoice2 模型转换  

## 模型架构
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

### 一、准备工作  

1. 下载模型  
在 python 中执行以下命令
```
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
```

2. 安装python包  
```
pip install -r model_convert/requirements.txt
``` 

### 二、生成calibration dataset  
1. 执行推理，保存所需的tensor数据  
```
python test.py
```
2. 将保存的tensor数据转换为npy，并打包为tar包
```
python gen_calib_data.py
```

### 三、导出 onnx  
1. 导出 flow encoder 和  flow decoder estimator  
```
bash export_flow.sh 
```
2. 导出 hift  
```
bash export_hift.sh
```
3. 导出llm decoder  
```
python  export_llm_decoder.py
```
4. 提取llm和flow中的embedding权重  
```
python extract_embedding.py
```

### 四、转换模型（onnx -> axmodel）  
1. 转换 flow encoder 和 flow decoder estimator  
```
bash build_flow.sh
```
将转换好的axmodel模型复制到当前目录下
2. 转换 hift  
```
bash build_hift.sh
```
将转换好的axmodel模型复制到当前目录下
3. 转换 llm  
```
model_convert/build_llm.sh
```

### 五、测试axmodel  
将当前文件夹下所有文件全部复制到开发板（推荐用nfs挂载）  
1. 在开发板上安装环境
```
pip3 install requirements.txt
```
2. 运行测试  
```
python3 test_ax.py
```