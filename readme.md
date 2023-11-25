# llm-webui
A Gradio web UI for Large Language Models(Finetune, RAG, Chat).
### Chat UI
![](https://github.com/wangermeng2021/llm-webui/blob/main/pics/UI_1.png)
### Training UI
![](https://github.com/wangermeng2021/llm-webui/blob/main/pics/UI_2.png)
### visualization UI
![](https://github.com/wangermeng2021/llm-webui/blob/main/pics/UI_3.png)
### RAG(Retrieval-augmented generation) UI
![](https://github.com/wangermeng2021/llm-webui/blob/main/pics/UI_4.png)

## Features
* Finetune:lora/qlora
* RAG(Retrieval-augmented generation):
    * Support txt/pdf/docx
    * Show retrieved chunks
    * Support finetuned model 
* Training tracking and visualization   
* Config prompt template through UI
* Support online and offline model
* Support online and offline dataset
## Install
Clone repo and install [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) in a
[**Python>=3.8.0**](https://www.python.org/) environment
```bash
git clone https://github.com/wangermeng2021/llm-webui
cd llm-webui
pip install -r requirements.txt
```

## Run
#### &nbsp;&nbsp;1. python main.py
#### &nbsp;&nbsp;2. Set huggingface hub token 
 > if you want to download Llama-2 from huggingface hub , you need to configure token:"Setting"->"Huggingface Hub Token"
