
import pandas as pd
import math
import numpy as np
import gc
import os,requests
from src.utils.common import login_huggingface
import subprocess,threading
from src.finetune.huggingface_inference import HuggingfaceInference
from src.finetune.llama_cpp_inference import LlamaCppInference
from src.rag.qa_with_rag import QAWithRAG
import time
import gradio as gr
import os
from src.utils.common import read_yaml,get_first_row_from_dataset,\
    get_runs_model_names_from_dir,get_hg_model_names_from_dir,get_hg_model_names_and_gguf_from_dir,validate_model_path,get_runs_models
from src.utils.chat_prompts import get_model_type,get_chat_history_prompt,get_model_prompt_template
from transformers.training_args import OptimizerNames
from huggingface_hub import hf_hub_download
from src.utils import  download_model
from pathlib import Path
import traceback
import numpy as np
import glob
import shutil
import torch
from src.finetune.qlora_trainer import QloraTrainer
from src.finetune.qlora_trainer import TRAINING_STATUS
from src.utils.download_huggingface_repo import download_model_wrapper,download_dataset_wrapper
import socket

# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:8889'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:8889'

LOCAL_HOST_IP = "0.0.0.0"
TENSORBOARD_URL = "http://" + LOCAL_HOST_IP + ":6006/"
INIT_DATASET_NAME = "test_python_code_instructions_5000_rows"

RAG_DATA_LIST_DROPDOWN = ""
TEXT_SPLITTER_DROPDOWN = ""
CHUNK_SIZE_SLIDER = 0
CHUNK_OVERLAP_SLIDER = -1
SEPARATORS_TEXTBOX = ""
EMBEDDING_MODEL_SOURCE_RADIO = ""
HUB_EMBEDDING_MODEL_NAMES_DROPDOWN = ""
LOCAL_EMBEDDING_MODEL_NAMES_DROPDOWN = ""
CHAT_MODEL_SOURCE_RADIO = ""
HUB_CHAT_MODEL_NAMES_DROPDOWN = ""
LOCAL_CHAT_MODEL_NAMES_DROPDOWN = ""
SEARCH_TOP_K_SLIDER = ""
SEARCH_SCORE_THRESHOLD_SLIDER = ""

training_ret_val = -1
error_msg = ""
current_running_model_name = ""
infer_model = None
stop_generation_status = False
chatbot_history=[]
chatbot_height = 500
rag_chatbot_history=[]
rag_stop_generation_status = False
qa_with_rag = QAWithRAG()
train_param_config = {}
train_param_config["dataset"]={}
train_param_config["model"]={}
train_param_config["training"]={}

model_zoo_config = {}
transformer_optimizer_list = []
model_context_window = 0
init_train_file_path = None
init_val_file_path = None
INIT_PREFIX1 = ""
INIT_PREFIX2 = ""
INIT_PREFIX3 = ""
INIT_PREFIX4 = ""
INIT_COL1_TEXT = ""
INIT_COL2_TEXT = ""
INIT_COL3_TEXT = ""
INIT_COL4_TEXT = ""

col_names = []
DATASET_FIRST_ROW = None
local_model_list = ""
local_model_root_dir = ""
base_model_names = []
training_base_model_names = []
embedding_model_names = []
base_model_context_window = []
local_dataset_list = []
local_dataset_root_dir = ""



def get_local_embedding_model_list():
    local_model_root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag", "embedding_models")
    local_model_root_files = os.listdir(local_model_root_dir)
    local_model_list = []
    for model_dir in local_model_root_files:
        if os.path.isdir(os.path.join(local_model_root_dir, model_dir)):
            local_model_list.append(model_dir)
    return local_model_list,local_model_root_dir
def get_local_model_list():
    local_model_root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    local_model_root_files = os.listdir(local_model_root_dir)
    local_model_list = []
    for model_dir in local_model_root_files:
        if os.path.isdir(os.path.join(local_model_root_dir, model_dir)):
            local_model_list.append(model_dir)
    return local_model_list,local_model_root_dir
def get_local_dataset_list():
    local_dataset_list = []
    local_dataset_root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    matched_dataset_file_path_list = glob.glob(os.path.join(local_dataset_root_dir,"**","dataset_infos.json"),recursive=False)
    for matched_file_path in matched_dataset_file_path_list:
        matched_pos1 = matched_file_path.rfind("datasets")
        matched_pos2 = matched_file_path.rfind("dataset_infos.json")
        local_dataset_list.append(matched_file_path[matched_pos1 + 9:matched_pos2-1])
    matched_dataset_file_path_list = glob.glob(os.path.join(local_dataset_root_dir,"**","dataset_dict.json"),recursive=False)
    for matched_file_path in matched_dataset_file_path_list:
        matched_pos1 = matched_file_path.rfind("datasets")
        matched_pos2 = matched_file_path.rfind("dataset_dict.json")
        local_dataset_list.append(matched_file_path[matched_pos1 + 9:matched_pos2-1])
    return local_dataset_list,local_dataset_root_dir
def start_tensorboard_server():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((LOCAL_HOST_IP, 6006))
        s.close()
    except Exception as e:
        tensorboard_cmd = f"tensorboard --logdir {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')}  --reload_multifile True"
        tensorboard_proc = subprocess.Popen(tensorboard_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            shell=True, close_fds=True)  # bufsize=0, close_fds=True

def init():
    global config_dict,transformer_optimizer_list,model_context_window,init_train_file_path,init_val_file_path
    global INIT_PREFIX1,INIT_COL1_TEXT,INIT_PREFIX2,INIT_COL2_TEXT,INIT_PREFIX3,INIT_COL3_TEXT,INIT_PREFIX4,INIT_COL4_TEXT,col_names,DATASET_FIRST_ROW
    global local_model_list,local_model_root_dir
    global base_model_names,base_model_context_window,embedding_model_names,training_base_model_names
    global local_dataset_list, local_dataset_root_dir

    start_tensorboard_server()

    model_zoo_config = read_yaml(os.path.join(os.path.dirname(os.path.abspath(__file__)),"config","model_zoo.yaml"))
    transformer_optimizer_list = list(vars(OptimizerNames)["_value2member_map_"].keys())

    #get dynamic context window from selected model
    model_context_window = [2048,1024,512]

    init_train_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", INIT_DATASET_NAME)

    DATASET_FIRST_ROW,split_list = get_first_row_from_dataset(init_train_file_path)

    col_names = list(DATASET_FIRST_ROW)
    col_names.insert(0,"")
    INIT_PREFIX1 = "<s>[INST] "
    INIT_PREFIX2 = "here are the inputs "
    INIT_PREFIX3 = " [/INST]"
    INIT_PREFIX4 = "</s>"
    INIT_COL1_TEXT = str(DATASET_FIRST_ROW[col_names[1]])
    INIT_COL2_TEXT = str(DATASET_FIRST_ROW[col_names[2]])
    INIT_COL3_TEXT = str(DATASET_FIRST_ROW[col_names[3]])
    INIT_COL4_TEXT = ""
    local_model_list,local_model_root_dir = get_local_model_list()
    base_model_names = [model_name for model_name in model_zoo_config["model_list"]]
    training_base_model_names = [model_name for model_name in base_model_names if not model_name.endswith(".gguf")]
    # base_model_context_window =  [model_name[1] for model_name in model_zoo_config["model_list"]]
    embedding_model_names = [model_name for model_name in model_zoo_config["embedding_model_list"]]
    local_dataset_list, local_dataset_root_dir = get_local_dataset_list()




with gr.Blocks(title="FINETUNE",css="#vertical_center_align_markdown { position:absolute; top:30%;background-color:white;} .white_background {background-color: #ffffff} .none_border {border: none;border-collapse:collapse;}") as demo:
    init()
    local_model_root_dir_textbox = gr.Textbox(label="", value=local_model_root_dir, visible=False)
    local_dataset_root_dir_textbox = gr.Textbox(label="",value=local_dataset_root_dir, visible=False)
    local_embedding_model_root_dir_textbox = gr.Textbox(label="", value=os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag", "embedding_models"), visible=False)
    local_chat_model_root_dir_textbox = gr.Textbox(label="", value=local_model_root_dir, visible=False)
    local_home_chat_model_root_dir_textbox = gr.Textbox(label="", value=local_model_root_dir, visible=False)
    session_state = gr.State(value={})
    # html = gr.HTML("<p  align='center';>llm-web-ui</p>",elem_id="header")

    with gr.Tab("Home"):
        with gr.Row():
            # with gr.Column(scale=4, min_width=1):
                with gr.Group():
                    gr.Markdown("## &nbsp;ChatBot", elem_classes="white_background")
                    with gr.Group():
                        gr.Markdown("### &nbsp;&nbsp;&nbsp;&nbsp;Chat Model", elem_classes="white_background")
                        local_home_chat_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
                        runs_model_root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
                        local_home_chat_model_names = get_hg_model_names_and_gguf_from_dir(local_home_chat_model_dir,
                                                                                      runs_model_root_dir)
                        home_chat_model_source_radio_choices = ["Download From Huggingface Hub",
                                                           f"From Local Dir(hg format:{local_home_chat_model_dir})"]
                        home_chat_model_source_radio = gr.Radio(home_chat_model_source_radio_choices,
                                                           label="Chat Model source", show_label=False,
                                                           value=home_chat_model_source_radio_choices[0],
                                                           interactive=True)
                    with gr.Row():
                        hub_home_chat_model_names_dropdown = gr.Dropdown(base_model_names,
                                                                    label=f"Chat Model", show_label=False,
                                                                    allow_custom_value=True,
                                                                    value=base_model_names[
                                                                        0] if base_model_names else None,
                                                                    interactive=True, scale=4, min_width=1)
                        local_home_chat_model_names_dropdown = gr.Dropdown(local_home_chat_model_names,
                                                                      label=f"Chat Model", show_label=False,
                                                                      value=local_home_chat_model_names[
                                                                          0] if local_home_chat_model_names else None,
                                                                      interactive=True, scale=4, min_width=1,
                                                                      visible=False)

                        download_hub_home_chat_model_names_btn = gr.Button("Download", scale=1)
                        stop_download_hub_home_chat_model_names_btn = gr.Button("Stop", scale=1, visible=False)
                        refresh_local_home_chat_model_names_btn = gr.Button("Refresh", scale=1, visible=False)
                        load_home_chat_model_btn = gr.Button("Load Model", scale=1, visible=True)
                        using_4bit_quantization_checkbox = gr.Checkbox(True, label="Using 4-bit quantization",
                                                                   interactive=True, visible=True,
                                                                   info="Less memory but slower", scale=1
                                                                   )

                    if validate_model_path(base_model_names[0])[0]:
                        download_hub_home_chat_model_status_markdown = gr.Markdown(
                            '<span style="color:green">&nbsp;&nbsp;&nbsp;&nbsp;This model has already been downloaded to local,click load model to run.</span>')
                    else:
                        download_hub_home_chat_model_status_markdown = gr.Markdown(
                            '<span style="color:red">&nbsp;&nbsp;&nbsp;&nbsp;This model has not been downloaded.</span>')
                    # home_chat_model_running_status_markdown = gr.Markdown(
                    #     '<span style="color:red">&nbsp;&nbsp;&nbsp;&nbsp;This model has not been downloaded.</span>')

                    with gr.Row():
                        chatbot = gr.Chatbot(value=[],bubble_full_width=False,rtl=False,layout="panel",height=chatbot_height,
                                             avatar_images=((os.path.join(os.path.abspath(''),"pics", "user1.png")), (os.path.join(os.path.abspath(''),"pics", "bot4.png"))),
                                             )
                    with gr.Row():

                            input_txtbox = gr.Textbox(
                                show_label=False,autofocus=True,
                                placeholder="Enter text and press enter",scale=3
                            )
                            generate_btn = gr.Button("Generate", scale=1)
                            stop_btn = gr.Button("Stop", scale=1)
                            # clear_btn = gr.Button("Clear",scale=1)



    with gr.Tab("Fine-Tuning"):
        with gr.Tabs() as tensorboard_tab:
            with gr.TabItem("Training", id=0):
                with gr.Row():
                    with gr.Column(scale=1, min_width=1):
                        with gr.Group():
                            gr.Markdown("## &nbsp;1.Training", elem_classes="white_background")
                            with gr.Group():
                                gr.Markdown("### &nbsp;1).Model", elem_classes="white_background")
                                with gr.Group():
                                    # gr.Markdown("<br> &nbsp;&nbsp;&nbsp; Base Model")
                                    base_model_source_radio_choices = ["Download From Huggingface Hub",
                                                                       f"From Local Dir(hg format:{local_model_root_dir})"]
                                    base_model_source_radio = gr.Radio(base_model_source_radio_choices,
                                                                       label="Base Model",
                                                                       value=base_model_source_radio_choices[0],
                                                                       interactive=True)
                                    with gr.Row(elem_classes="white_background"):
                                        base_model_name_dropdown = gr.Dropdown(training_base_model_names,
                                                                               label="Model Name", value=training_base_model_names[0] if training_base_model_names else None,
                                                                               interactive=True, visible=True, scale=5,
                                                                               allow_custom_value=True)
                                        download_local_model_btn = gr.Button("Download", scale=1, visible=True)
                                        stop_download_local_model_btn = gr.Button("Stop", scale=1, visible=False)
                                        # model_download_status = gr.Markdown("<div id='vertical_center_align_markdown'><p style='text-align: center;'>Not downloaded</p></div>", elem_classes="white_background",scale=1,full_width=True,visible=False)

                                    if validate_model_path(training_base_model_names[0])[0]:
                                        download_model_status_markdown = gr.Markdown('<span style="color:green">&nbsp;&nbsp;&nbsp;&nbsp;This model has already been downloaded to local.</span>')
                                    else:
                                        download_model_status_markdown = gr.Markdown('<span style="color:red">&nbsp;&nbsp;&nbsp;&nbsp;This model has not been downloaded.</span>')

                                    with gr.Row():
                                        # local_home_chat_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
                                        # runs_model_root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
                                        # local_model_list = get_hg_model_names_and_gguf_from_dir(local_home_chat_model_dir,runs_model_root_dir)
                                        local_model_list = get_hg_model_names_from_dir(os.path.dirname(os.path.abspath(__file__)), "models")
                                        local_model_dropdown = gr.Dropdown(local_model_list, label="Local Model",
                                                                           info="",
                                                                           value=local_model_list[0] if len(local_model_list) > 0 else None,
                                                                           interactive=True,
                                                                           elem_classes="white_background", scale=5,
                                                                           visible=False)
                                        refresh_local_model_list_btn = gr.Button("Refresh", scale=1, visible=False)
                                fine_tuning_type_dropdown = gr.Dropdown(["QLoRA", "LoRA"],
                                                                        label="Fine-Tuning Type", info="",
                                                                        value="QLoRA", interactive=True)
                                with gr.Group():
                                    with gr.Row(elem_classes="white_background"):
                                        # gr.Markdown("###  &nbsp;&nbsp;&nbsp; LoRA Config", elem_classes="white_background")
                                        lora_r_list = [str(ri) for ri in range(8, 65, 8)]
                                        lora_r_slider = gr.Slider(8, 64, value=8, step=8, label="lora_r",
                                                                  interactive=True)
                                        # lora_r_dropdown = gr.Dropdown(lora_r_list,label="lora_r", value=lora_r_list[0],interactive=True,allow_custom_value=True)

                                        lora_alpha_slider = gr.Slider(8, 96, value=32, step=8, label="lora_alpha",
                                                                      interactive=True)
                                        # lora_alpha_list = [str(ri) for ri in range(8, 97, 8)]
                                        # lora_alpha_dropdown = gr.Dropdown(lora_alpha_list,label="lora_alpha", value=lora_alpha_list[3],interactive=True,allow_custom_value=True)
                                    with gr.Row(elem_classes="white_background"):
                                        lora_dropout_slider = gr.Slider(0, 1, value=0.05, step=0.01,
                                                                        label="lora_dropout", interactive=True)
                                        lora_bias_dropdown = gr.Dropdown(["none", "all", "lora_only"],
                                                                         label="lora_bias", info="", value="none",
                                                                         interactive=True)

                            with gr.Group():
                                gr.Markdown("### &nbsp;2).Dataset",elem_classes="white_background")

                                dataset_source_radio_choices = ["Download From Huggingface Hub",
                                                                   f"From Local HG Dataset In {local_dataset_root_dir})"]
                                dataset_source_radio = gr.Radio(dataset_source_radio_choices, label="Dataset Source",
                                                                   value=dataset_source_radio_choices[1], interactive=True)
                                with gr.Row(equal_height=True):
                                    hg_dataset_path_textbox = gr.Textbox(label="Dataset Name:",elem_classes="none_border",visible=False, interactive=True, scale=4,
                                                                         value="iamtarun/python_code_instructions_18k_alpaca")
                                    download_local_dataset_btn = gr.Button("Download", scale=1, visible=False)
                                    stop_download_local_dataset_btn = gr.Button("Stop", scale=1, visible=False)
                                download_dataset_status_markdown = gr.Markdown('')
                                with gr.Row():
                                        hg_train_dataset_dropdown = gr.Dropdown(["train"], label="Train set", info="", interactive=False,visible=False,
                                                                           elem_classes="white_background", scale=1,value="train")
                                        hg_val_dataset_dropdown = gr.Dropdown([], label="Val set", info="", interactive=False,visible=False,
                                                                           elem_classes="white_background", scale=1)

                                with gr.Row():
                                    local_dataset_list.pop(
                                        local_dataset_list.index(INIT_DATASET_NAME))
                                    local_dataset_list.insert(0, INIT_DATASET_NAME)
                                    local_train_path_dataset_dropdown = gr.Dropdown(local_dataset_list, label="Train Dataset", info="",
                                                                       value=local_dataset_list[0] if len(local_dataset_list)>0 else None, interactive=True,
                                                                       elem_classes="white_background", scale=5, visible=True)
                                    refresh_local_train_path_dataset_list_btn = gr.Button("Refresh", scale=1, visible=True)
                                with gr.Row():
                                        local_train_dataset_dropdown = gr.Dropdown(["train"], label="Train set", info="", interactive=True,
                                                                           elem_classes="white_background", scale=1,value="train",visible=True)
                                        local_val_dataset_dropdown = gr.Dropdown([], label="Val set", info="", interactive=True,
                                                                           elem_classes="white_background", scale=1,visible=True)

                                with gr.Group(elem_classes="white_background"):
                                    # gr.Markdown("<h4><br> &nbsp;&nbsp;Prompt Template: (Prefix1 + ColumnName1 + Prefix2 + ColumnName2)</h4>",elem_classes="white_background")
                                    gr.Markdown("<br> &nbsp;&nbsp;&nbsp;&nbsp;**Prompt Template: (Prefix1+ColumnName1+Prefix2+ColumnName2+Prefix3+ColumnName3+Prefix4+ColumnName4)**",elem_classes="white_background")
                                    gr.Markdown(
                                        "<span> &nbsp;&nbsp;&nbsp;&nbsp;**Note**:&nbsp;&nbsp;Llama2/Mistral Chat Template:<s\>[INST] instruction+input [/INST] output</s\> </span>",elem_classes="white_background")
                                    # using_llama2_chat_template_checkbox = gr.Checkbox(True, label="Using Llama2/Mistral chat template",interactive=True,visible=False)
                                    with gr.Row(elem_classes="white_background"):
                                        # prompt_template
                                        prefix1_textbox = gr.Textbox(label="Prefix1:",value=INIT_PREFIX1,lines=2,interactive=True,elem_classes="white_background")
                                        datatset_col1_dropdown = gr.Dropdown(col_names, label="ColumnName1:", info="",value=col_names[1],interactive=True,elem_classes="white_background")
                                        prefix2_textbox = gr.Textbox(label="Prefix2:",value=INIT_PREFIX2,lines=2,interactive=True,elem_classes="white_background")
                                        datatset_col2_dropdown = gr.Dropdown(col_names, label="ColumnName2:", info="",value=col_names[2],interactive=True,elem_classes="white_background")
                                    with gr.Row(elem_classes="white_background"):
                                        prefix3_textbox = gr.Textbox(label="Prefix3:",value=INIT_PREFIX3,lines=2,interactive=True,elem_classes="white_background")
                                        datatset_col3_dropdown = gr.Dropdown(col_names, label="ColumnName3:", info="",value=col_names[3],interactive=True,elem_classes="white_background")
                                        prefix4_textbox = gr.Textbox(label="Prefix4:",value=INIT_PREFIX4,lines=2,interactive=True,elem_classes="white_background")
                                        datatset_col4_dropdown = gr.Dropdown(col_names, label="ColumnName4:", info="",value=col_names[0],interactive=True,elem_classes="white_background")
                                    # print("")
                                    prompt_sample = INIT_PREFIX1 + INIT_COL1_TEXT + INIT_PREFIX2 + INIT_COL2_TEXT + INIT_PREFIX3 + INIT_COL3_TEXT + INIT_PREFIX4 + INIT_COL4_TEXT
                                    prompt_sample_textbox = gr.Textbox(label="Prompt Sample:",interactive=False,value=prompt_sample,lines=4)
                                    max_length_dropdown = gr.Dropdown(["Model Max Length"]+model_context_window, label="Max Length",value="Model Max Length", interactive=True,allow_custom_value=True)


                            with gr.Group():
                                gr.Markdown("### &nbsp;3).Training Arguments",elem_classes="white_background")
                                with gr.Row(elem_classes="white_background"):
                                    epochs_slider = gr.Slider(1, 100, value=10, step=1, label="Epochs", interactive=True)
                                    # epochs_dropdown = gr.Dropdown([1]+[bi for bi in range(10,101,10)], label="Epochs",value=1, interactive=True,allow_custom_value=True)
                                    batch_size_list = [1,2,3]+[bi for bi in range(4,32+1,4)]
                                    batch_size_slider = gr.Slider(1, 100, value=1, step=1, label="Batch Size", interactive=True)
                                    # batch_size_dropdown = gr.Dropdown(batch_size_list,label="Batch Size", info="",value=batch_size_list[0],interactive=True,allow_custom_value=True)
                                    # learning_rate_textbox = gr.Textbox(label="Learning Rate", value=2e-4,interactive=True)
                                with gr.Row(elem_classes="white_background"):
                                    learning_rate_slider = gr.Slider(0, 0.01, value=2e-4, step=0.0001, label="Learning Rate", interactive=True)
                                    warmup_steps_slider = gr.Slider(0, 400, value=100, step=10, label="Warmup Steps",
                                                                    interactive=True)
                                with gr.Row(elem_classes="white_background"):
                                    optimizer_dropdown = gr.Dropdown(transformer_optimizer_list, label="Optimizer", info="",
                                                                     value=transformer_optimizer_list[1], interactive=True)
                                    lr_scheduler_list = ["linear","cosine","cosine_with_hard_restarts","polynomial_decay","constant","constant_with_warmup","inverse_sqrt","reduce_on_plateau"]
                                    lr_scheduler_type_dropdown = gr.Dropdown(lr_scheduler_list, label="LR Scheduler Type", info="",
                                                                     value=lr_scheduler_list[0], interactive=True)
                                with gr.Row(elem_classes="white_background"):
                                    early_stopping_patience_slider = gr.Slider(0, 50+1, value=0, step=5, label="Early Stopping Patience",
                                                                    interactive=True)
                                    gradient_accumulation_steps_slider = gr.Slider(1, 50, value=1, step=1,
                                                                                   label="Gradient Accumulation Steps")
                                with gr.Row(elem_classes="white_background"):
                                    eval_steps_slider = gr.Slider(0, 1000, value=100, step=100, label="eval_steps", interactive=True)
                                    gradient_checkpointing_checkbox = gr.Checkbox(True,label="Gradient Checkpointing",interactive=True)
                            train_btn = gr.Button("Start Training")

                    with gr.Column(scale=1, min_width=1):
                        with gr.Group():
                            gr.Markdown("## &nbsp;2.Test",elem_classes="white_background")
                            training_runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')
                            run_names = os.listdir(training_runs_dir)
                            run_names.sort(key=lambda file:os.path.getmtime(os.path.join(training_runs_dir,file)))
                            runs_output_model = []
                            for run_name in run_names:
                                run_name_dir = os.path.join(training_runs_dir,run_name)
                                run_output_model = os.path.join(run_name_dir,"output_model")
                                if os.path.exists(run_output_model):
                                    run_output_model_names = os.listdir(run_output_model)
                                    for run_output_model_name in run_output_model_names:
                                        if run_output_model_name.find("merged_")>=0:
                                            runs_output_model.append(os.path.join(run_name,"output_model",run_output_model_name, "ori"))
                            runs_output_model = runs_output_model[::-1]
                            runs_output_model_dropdown = gr.Dropdown(runs_output_model, label="runs_output_model",
                                                             value=runs_output_model[0] if runs_output_model else None, interactive=True)
                            gr.Markdown("")
                            gr.Markdown(
                                "<span> &nbsp;&nbsp;&nbsp;&nbsp;**Note**:&nbsp;&nbsp;Llama2/Mistral Chat Template:<s\>[INST] instruction+input [/INST] output</s\> </span>",
                                elem_classes="white_background")
                            with gr.Row():
                                test_input_textbox = gr.Textbox(label="Input:", interactive=True, value="", lines=4,
                                                        scale=4)
                                generate_text_btn = gr.Button("Generate",scale=1)
                                finetune_test_using_4bit_quantization_checkbox = gr.Checkbox(True, label="Using 4-bit quantization",
                                                                               interactive=True, visible=True,
                                                                               info="Less memory but slower", scale=1
                                                                               )
                            # test_prompt = gr.Textbox(label="Prompt:", interactive=False, lines=2, scale=1)
                            test_output = gr.Textbox(label="Output:", interactive=False,lines=4, scale=1)


                            # def change_test_input_textbox(test_prefix1_textbox,test_input_textbox,test_prefix2_textbox):
                            #     return gr.update(value=test_prefix1_textbox+test_input_textbox+test_prefix2_textbox)
                            # test_input_textbox.change(change_test_input_textbox,[test_prefix1_textbox,test_input_textbox,test_prefix2_textbox],test_prompt)


                        with gr.Group():
                            gr.Markdown("## &nbsp;3.Quantization",elem_classes="white_background")
                            with gr.Row():
                                quantization_type_list = ["gguf"]
                                quantization_type_dropdown = gr.Dropdown(quantization_type_list, label="Quantization Type",value=quantization_type_list[0], interactive=True,scale=3)
                                local_quantization_dataset_dropdown = gr.Dropdown(local_dataset_list, label="Dataset for quantization",
                                                                           value=local_dataset_list[0] if len(
                                                                               local_dataset_list) > 0 else None,
                                                                           interactive=True,
                                                                           elem_classes="white_background", scale=7,
                                                                           visible=False)
                                refresh_local_quantization_dataset_btn = gr.Button("Refresh", scale=2, visible=False)
                                def click_refresh_local_quantization_dataset_btn():
                                    local_dataset_list, _ = get_local_dataset_list()
                                    return gr.update(choices=local_dataset_list,
                                                     value=local_dataset_list[0] if len(local_dataset_list) > 0 else "")
                                refresh_local_quantization_dataset_btn.click(click_refresh_local_quantization_dataset_btn,[],local_quantization_dataset_dropdown)


                            with gr.Row():
                                training_runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')
                                run_names = os.listdir(training_runs_dir)
                                run_names.sort(key=lambda file: os.path.getmtime(os.path.join(training_runs_dir, file)))
                                runs_output_model = []
                                for run_name in run_names:
                                    run_name_dir = os.path.join(training_runs_dir, run_name)
                                    run_output_model = os.path.join(run_name_dir, "output_model")
                                    if os.path.exists(run_output_model):
                                        run_output_model_names = os.listdir(run_output_model)
                                        for run_output_model_name in run_output_model_names:
                                            if run_output_model_name.find("merged_") >= 0:
                                                runs_output_model.append(
                                                    os.path.join(run_name, "output_model", run_output_model_name,
                                                                 "ori"))
                                runs_output_model = runs_output_model[::-1]
                                quantization_runs_output_model_dropdown = gr.Dropdown(runs_output_model,
                                                                                    label="runs_output_model",
                                                                                    value=runs_output_model[
                                                                                        0] if runs_output_model else None,
                                                                                    interactive=True, scale=6)

                                quantize_btn = gr.Button("Quantize", scale=1,visible=False)
                            if runs_output_model:
                                model_name = runs_output_model[0].split(os.sep)[-2].split('_')[-1]
                                quantized_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs',
                                                                   os.sep.join(runs_output_model[0].split(os.sep)[0:-1]),
                                                                   "quantized_" + quantization_type_list[0] + "_" + model_name)
                                if not os.path.exists(quantized_model_dir):
                                    os.makedirs(quantized_model_dir)
                                quantization_logging_markdown = gr.Markdown("")
                                gguf_quantization_markdown0 = gr.Markdown("### &nbsp;&nbsp;&nbsp;&nbsp;GGUF Quantization Instruction:", elem_classes="white_background", visible=True)
                                gguf_quantization_markdown1 = gr.Markdown('''&nbsp;&nbsp;&nbsp;&nbsp;1.Follow the instructions in the llama.cpp to generate a GGUF:[https://github.com/ggerganov/llama.cpp#prepare-data--run](https://github.com/ggerganov/llama.cpp#prepare-data--run),<span style="color:red">&nbsp;&nbsp;Q4_K_M is recommend</span>''',visible=True)
                                if runs_output_model:
                                    gguf_quantization_markdown2 = gr.Markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;2.Convert {runs_output_model[0]} to gguf model",visible=True)
                                else:
                                    gguf_quantization_markdown2 = gr.Markdown(
                                        f"", visible=True)
                                gguf_quantization_markdown3 = gr.Markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;3.Deploy gguf model", visible=False)
                            else:

                                quantization_logging_markdown = gr.Markdown("")
                                gguf_quantization_markdown0 = gr.Markdown("### &nbsp;&nbsp;&nbsp;&nbsp;GGUF Quantization Instruction:", elem_classes="white_background", visible=True)
                                gguf_quantization_markdown1 = gr.Markdown('''''',visible=True)
                                gguf_quantization_markdown2 = gr.Markdown(f"",visible=True)
                                gguf_quantization_markdown3 = gr.Markdown(f"", visible=True)

                        with gr.Group(visible=False):
                            gr.Markdown("## &nbsp;4.Deploy",elem_classes="white_background")
                            with gr.Row():
                                deployment_framework_dropdown = gr.Dropdown(["TGI","llama-cpp-python"], label="Deployment Framework",value="TGI", interactive=True)
                            with gr.Row():
                                training_runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')
                                run_names = os.listdir(training_runs_dir)
                                run_names.sort(key=lambda file: os.path.getmtime(os.path.join(training_runs_dir, file)))
                                # ori_model_runs_output_model = []
                                tgi_model_format_runs_output_model = []
                                gguf_model_format_runs_output_model = []
                                for run_name in run_names:
                                    run_name_dir = os.path.join(training_runs_dir, run_name)
                                    run_output_model = os.path.join(run_name_dir, "output_model")
                                    if os.path.exists(run_output_model):
                                        run_output_model_names = os.listdir(run_output_model)
                                        for run_output_model_name in run_output_model_names:
                                            model_bin_path = os.path.exists(
                                                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs',
                                                             run_name, "output_model", run_output_model_name, "ori",
                                                             "pytorch_model.bin"))
                                            if run_output_model_name.find("merged_") >= 0 and model_bin_path:
                                                tgi_model_format_runs_output_model.append(
                                                    os.path.join(run_name, "output_model", run_output_model_name, "ori"))

                                                gptq_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs',run_name, "output_model", run_output_model_name, "quantized_gptq_"+run_output_model_name.split('_')[-1],
                                                             "pytorch_model.bin")
                                                if os.path.exists(gptq_model_path):
                                                    tgi_model_format_runs_output_model.append(os.path.join(run_name, "output_model", run_output_model_name, "quantized_gptq_"+run_output_model_name.split('_')[-1]))
                                                gguf_model_dir = os.path.join(
                                                    os.path.dirname(os.path.abspath(__file__)), 'runs', run_name,
                                                    "output_model", run_output_model_name,
                                                    "quantized_gguf_" + run_output_model_name.split('_')[-1])
                                                if os.path.exists(gguf_model_dir):
                                                    gguf_model_names = os.listdir(gguf_model_dir)
                                                    for gguf_model_name in gguf_model_names:
                                                        if gguf_model_name.split('.')[-1] == "gguf":
                                                            gguf_model_format_runs_output_model.append(
                                                                os.path.join(run_name, "output_model",
                                                                             run_output_model_name, "quantized_gguf_" +
                                                                             run_output_model_name.split('_')[-1],
                                                                             gguf_model_name))

                                tgi_model_format_runs_output_model = tgi_model_format_runs_output_model[::-1]
                                gguf_model_format_runs_output_model = gguf_model_format_runs_output_model[::-1]

                                deployment_runs_output_model_dropdown = gr.Dropdown(tgi_model_format_runs_output_model, label="runs_output_model",
                                                                         value=tgi_model_format_runs_output_model[
                                                                             0] if tgi_model_format_runs_output_model else None,
                                                                         interactive=True,scale=6)
                                refresh_deployment_runs_output_model_btn = gr.Button("Refresh", scale=1, visible=True)


                            if tgi_model_format_runs_output_model:
                                model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs',
                                                         os.path.dirname(tgi_model_format_runs_output_model[0]))
                                model_name = os.path.basename(tgi_model_format_runs_output_model[0])
                                if model_name.rfind("quantized_gptq_") >= 0:
                                    run_server_value = f'''docker run --gpus all --shm-size 1g -p 8080:80 -v {model_dir}:/data ghcr.io/huggingface/text-generation-inference:latest --model-id /data/{model_name} --quantize gptq'''
                                else:
                                    run_server_value = f'''docker run --gpus all --shm-size 1g -p 8080:80 -v {model_dir}:/data ghcr.io/huggingface/text-generation-inference:latest --model-id /data/{model_name}'''

                                run_server_script_textbox = gr.Textbox(label="Run Server:", interactive=False,lines=2, scale=1,value=run_server_value)
                                run_client_value = '''Command-Line Interface(CLI):\ncurl 127.0.0.1:8080/generate -X POST  -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":20}}' -H 'Content-Type: application/json'\n\nPython:\nfrom huggingface_hub import InferenceClient \nclient = InferenceClient(model="http://127.0.0.1:8080")\noutput = client.text_generation(prompt="What is Deep Learning?",max_new_tokens=512)
                                '''
                                run_client_script_textbox = gr.Textbox(label="Run Client:", interactive=False, lines=6,scale=1,value=run_client_value)
                            else:
                                run_server_script_textbox = gr.Textbox(label="Run Server:", interactive=False,lines=2, scale=1,value="")
                                run_client_script_textbox = gr.Textbox(label="Run Client:", interactive=False, lines=6,
                                                                       scale=1, value="")

                            # deploy_llm_code = gr.Code(code_str, language="shell", lines=5, label="Install Requirements:")
                            install_requirements_value = '''
                            ### &nbsp;&nbsp; 1.install docker 
                            ### &nbsp;&nbsp; 2.Install NVIDIA Container Toolkit
                            <h4> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.1 Configure the repository: </h4>
                            <p> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
  && \
    sudo apt-get update </p>
                                <h4> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.2 Install the NVIDIA Container Toolkit packages: </h4> 
                                <p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; sudo apt-get install -y nvidia-container-toolkit </p>
                                '''
                            with gr.Accordion("Install Requirements",open=False) as install_requirements_accordion:
                                install_requirements_markdown = gr.Markdown(install_requirements_value)


                            run_llama_cpp_python_code = gr.Code("", language="python", lines=10, label="run_model_using_llama_cpp_python.py",visible=False)
                            # run_script_textbox = gr.Textbox(label="Install Requirements:", interactive=False, scale=1,value=install_requirements_value)
                            #dependencies


            with gr.TabItem("Tensorboard", id=1) as fdddd:

                # training_log_markdown = gr.Markdown('',every=mytestfun)
                with gr.Row():
                    # training_log_textbox = gr.Textbox(label="logging:",value="", interactive=True, lines=2, scale=1)
                    with gr.Group():
                        training_log_markdown = gr.Markdown('')
                        stop_training_btn = gr.Button("Stop Training")
                    training_runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')
                    run_names = os.listdir(training_runs_dir)
                    run_names = [run_name for run_name in run_names if os.path.isdir(os.path.join(training_runs_dir,run_name))]
                    run_names.sort(key=lambda f: os.path.getmtime(os.path.join(training_runs_dir, f)))
                    # print("dddddddd:",run_names)
                    with gr.Group():
                        # with gr.Row():
                            training_runs_dropdown = gr.Dropdown(run_names, label="Training Runs",value=run_names[0] if run_names else None, interactive=True, scale=1)
                            delete_text_btn = gr.Button("Delete Run", scale=1)


                iframe = f'<iframe src={TENSORBOARD_URL} style="border:none;height:1024px;width:100%">'
                tensorboard_html = gr.HTML(iframe)


    with gr.Tab("RAG"):
        with gr.Row():
            with gr.Column(scale=4, min_width=1):
                with gr.Group():
                    gr.Markdown("## &nbsp;ChatBot", elem_classes="white_background")

                    rag_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rag', 'data')
                    matched_file_list = []
                    supported_doc_type = ["*.pdf","*.txt","*.docx"]
                    for doc_type in supported_doc_type:
                        matched_file_list += glob.glob(os.path.join(rag_data_dir, doc_type), recursive=False)
                    matched_file_list.sort(key=lambda file: os.path.getmtime(file),reverse=True)
                    matched_file_name_list = []
                    for matched_file in  matched_file_list:
                        matched_file_name_list.append(os.path.basename(matched_file))

                    # chat_data_source_radio_choices = ["Chat With Document",
                    #                                    f"Chat With Image"]
                    gr.Markdown("### &nbsp;Chat With Document", elem_classes="white_background")
                    # chat_data_source_radio = gr.Radio(chat_data_source_radio_choices,
                    #                                    label="",
                    #                                    value=chat_data_source_radio_choices[0],
                    #                                    interactive=True)
                    with gr.Row():
                            rag_data_list_dropdown = gr.Dropdown(matched_file_name_list, label=f"Local Documents In {rag_data_dir}",
                                                                     value=matched_file_name_list[0] if matched_file_name_list else None,
                                                                     interactive=True,scale=4, min_width=1)
                            refresh_rag_data_list_btn = gr.Button("Refresh", scale=1, min_width=1)

                        # if not current_running_model_name:
                        #     model_running_status_markdown = gr.Markdown(f"<span style='color:red'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;No modelis running!</span>")
                        # else:
                        #     model_running_status_markdown = gr.Markdown(f"<span style='color:green'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Model is runing:{current_running_model_name}.</span>")

                    def click_refresh_rag_data_list_btn():
                        rag_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rag', 'data')
                        matched_file_list = []
                        supported_doc_type = ["*.pdf", "*.txt", "*.docx"]
                        for doc_type in supported_doc_type:
                            matched_file_list += glob.glob(os.path.join(rag_data_dir, doc_type), recursive=False)
                        matched_file_list.sort(key=lambda file: os.path.getmtime(file), reverse=True)
                        matched_file_name_list = []
                        for matched_file in matched_file_list:
                            matched_file_name_list.append(os.path.basename(matched_file))
                        return gr.update(choices=matched_file_name_list,value=matched_file_name_list[0] if matched_file_name_list else None)
                    refresh_rag_data_list_btn.click(click_refresh_rag_data_list_btn,[],rag_data_list_dropdown)



                    # def update_model_running_status():
                    #     return gr.update(value=f"<span style='color:red'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{current_running_model_name} is runing!.</span>")
                    #
                    # load_model_btn.click(click_load_model_btn,model_list_dropdown,[model_list_dropdown]).success(update_model_running_status,[],model_running_status_markdown)
                    with gr.Row():
                        rag_chatbot = gr.Chatbot(value=[],bubble_full_width=False,rtl=False,layout="panel",height=chatbot_height,
                                             avatar_images=((os.path.join(os.path.abspath(''),"pics", "user1.png")), (os.path.join(os.path.abspath(''),"pics", "bot4.png"))),
                                             )
                    with gr.Row():
                        rag_input_txtbox = gr.Textbox(
                                show_label=False,autofocus=True,
                                placeholder="Enter text and press enter",scale=6)
                        rag_generate_btn = gr.Button("Generate", scale=1)
                        rag_stop_btn = gr.Button("Stop", scale=1)
                        # rag_clear_btn = gr.Button("Clear", scale=1)
                    rag_model_running_status_markdown = gr.Markdown(
                        f"### &nbsp;&nbsp;Retrieved Document Chunks",visible=True)
                    # retrieved_document_chunks_markdown = gr.Markdown(
                    #     f"### &nbsp;&nbsp;Retrieved Document Chunks",visible=True)
                    retrieved_document_chunks_dataframe = gr.Dataframe(
                        headers=["ID", "Chunk"],
                        datatype=["str", "str"],
                        show_label=False,
                        value=None
                    )
            with gr.Column(scale=4, min_width=1):
                with gr.Group():
                    gr.Markdown("## &nbsp;Setting", elem_classes="white_background")
                    with gr.Group():
                        with gr.Group():
                            gr.Markdown("### &nbsp;&nbsp;1.Chunking", elem_classes="white_background")
                            with gr.Row():
                                text_splitter_dropdown = gr.Dropdown(["RecursiveCharacterTextSplitter"],
                                                                  label=f"Text Splitter",
                                                                  value="RecursiveCharacterTextSplitter",
                                                                  interactive=True, scale=1, min_width=1)
                            with gr.Row():
                                chunk_size_slider = gr.Slider(32, 1024, value=256, step=32, label="Chunk Size",
                                                              interactive=True, scale=1)
                                chunk_overlap_slider = gr.Slider(0, 500, value=20, step=10, label="Chunk Overlap",
                                                                 interactive=True)
                                Separators_textbox = gr.Textbox(label="Separators",
                                                                value='''["\n\n", "\n", ".", " ", ""]''',
                                                                interactive=True,visible=False)
                        with gr.Group():
                            gr.Markdown("### &nbsp;&nbsp;2.Vector Store Retriever", elem_classes="white_background") #
                            local_embedding_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"rag","embedding_models")
                            local_embedding_model_names = get_hg_model_names_from_dir(local_embedding_model_dir,"embedding_models")
                            embedding_model_source_radio_choices = ["Download From Huggingface Hub",
                                                               f"From Local Dir(hg format:{local_embedding_model_dir})"]
                            embedding_model_source_radio = gr.Radio(embedding_model_source_radio_choices,
                                                               label="Embedding Model Source",
                                                               value=embedding_model_source_radio_choices[0],
                                                               interactive=True)
                            with gr.Row():
                                hub_embedding_model_names_dropdown = gr.Dropdown(embedding_model_names,
                                                                  label=f"",show_label=False,
                                                                  value=embedding_model_names[0] if embedding_model_names else None,
                                                                  interactive=True, scale=4, min_width=1)
                                download_hub_embedding_model_names_btn = gr.Button("Download", scale=1)
                                stop_download_hub_embedding_model_names_btn = gr.Button("Stop", scale=1, visible=False)

                                local_embedding_model_names_dropdown = gr.Dropdown(local_embedding_model_names,
                                                                  label=f"Embedding Model",show_label=False,
                                                                  value=local_embedding_model_names[0] if local_embedding_model_names else None,
                                                                  interactive=True, scale=4, min_width=1,visible=False)
                                refresh_local_embedding_model_names_btn = gr.Button("Refresh", scale=1,visible=False)

                            # model_config_path1 = os.path.join(local_embedding_model_dir,
                            #                                  embedding_model_names[0], "pytorch_model.bin")
                            # model_config_path2 = os.path.join(local_embedding_model_dir,
                            #                                  embedding_model_names[0], "model.safetensors")
                            model_config_path = os.path.join(local_embedding_model_dir,
                                                             embedding_model_names[0], "config.json")
                            if os.path.exists(model_config_path):
                                download_hub_embedding_model_status_markdown = gr.Markdown(
                                    '<span style="color:green">&nbsp;&nbsp;&nbsp;&nbsp;This model has already been downloaded to local.</span>')
                            else:
                                download_hub_embedding_model_status_markdown = gr.Markdown(
                                    '<span style="color:red">&nbsp;&nbsp;&nbsp;&nbsp;This model has not been downloaded.</span>')
                            with gr.Row():
                                search_top_k_slider = gr.Slider(1, 10, value=3, step=1, label="Search Top K", interactive=True)
                                search_score_threshold_slider = gr.Slider(0, 1, value=0.5, step=0.1, label="Search Score Threshold",interactive=True)

                        with gr.Group():
                            gr.Markdown("### &nbsp;&nbsp;3.Chat Model", elem_classes="white_background")
                            local_chat_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"models")
                            runs_model_root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
                            # local_chat_model_names = get_hg_model_names_from_dir(local_chat_model_dir)
                            local_chat_model_names = get_hg_model_names_and_gguf_from_dir(local_chat_model_dir,runs_model_root_dir)
                            chat_model_source_radio_choices = ["Download From Huggingface Hub",
                                                               f"From Local Dir(hg format:{local_chat_model_dir})"]
                            chat_model_source_radio = gr.Radio(chat_model_source_radio_choices,
                                                               label="Chat Model source",show_label=False,
                                                               value=chat_model_source_radio_choices[0],
                                                               interactive=True)
                            with gr.Row():
                                hub_chat_model_names_dropdown = gr.Dropdown(base_model_names,
                                                                  label=f"Chat Model",show_label=False,allow_custom_value=True,
                                                                  value=base_model_names[0] if base_model_names else None,
                                                                  interactive=True, scale=4, min_width=1)
                                download_hub_chat_model_names_btn = gr.Button("Download", scale=1)
                                stop_download_hub_chat_model_names_btn = gr.Button("Stop", scale=1, visible=False)
                                local_chat_model_names_dropdown = gr.Dropdown(local_chat_model_names,
                                                                  label=f"Chat Model",show_label=False,
                                                                  value=local_chat_model_names[0] if local_chat_model_names else None,
                                                                  interactive=True, scale=4, min_width=1,visible=False)
                                refresh_local_chat_model_names_btn = gr.Button("Refresh", scale=1,visible=False)
                                rag_using_4bit_quantization_checkbox = gr.Checkbox(True, label="Using 4-bit quantization",
                                                                               interactive=True, visible=True,
                                                                               info="Less memory but slower", scale=1
                                                                               )
                            if validate_model_path(base_model_names[0])[0]:
                                download_hub_chat_model_status_markdown = gr.Markdown(
                                    '<span style="color:green">&nbsp;&nbsp;&nbsp;&nbsp;This model has already been downloaded to local.</span>')
                            else:
                                download_hub_chat_model_status_markdown = gr.Markdown(
                                    '<span style="color:red">&nbsp;&nbsp;&nbsp;&nbsp;This model has not been downloaded.</span>')

    with gr.Tab("Setting"):
        # with gr.Column(scale=4, min_width=1):
            with gr.Group():
                gr.Markdown("## &nbsp;Setting", elem_classes="white_background")
                with gr.Group():
                    with gr.Row():
                        max_new_tokens_slider = gr.Slider(1, 4096, value=256, step=0.1, label="Max New Tokens",
                                                          interactive=True)
                        temperature_slider = gr.Slider(0, 5, value=1, step=0.1, label="Temperature",
                                                       interactive=True)
                    with gr.Row():
                        top_k_slider = gr.Slider(1, 100, value=50, step=1, label="Top_k",
                                                 interactive=True)
                        top_p_slider = gr.Slider(0, 1, value=1, step=0.1, label="Top_p",
                                                 interactive=True)
                    with gr.Row():
                        repeat_penalty_slider = gr.Slider(1, 5, value=1, step=0.1, label="Repeat Penalty", interactive=True)
                    with gr.Row():
                        chat_history_window_slider = gr.Slider(1, 20, value=3, step=1, label="Chat History Window",
                                                               interactive=True)
                        low_cpu_mem_usage_checkbox = gr.Checkbox(False, label="Low Cpu Mem Usage",interactive=True,visible=False)
                        Huggingface_hub_token = gr.Textbox(label="Huggingface Hub Token", value="")


    def check_local_model_or_dataset_is_empty1(base_model_name_dropdown,Huggingface_hub_token):
        if len(base_model_name_dropdown.strip()) == 0:
            raise gr.Error("Name is empty!")
        try:
            login_huggingface(Huggingface_hub_token,base_model_name_dropdown)
        except Exception as e:
            raise gr.Error(e)
    def check_local_model_or_dataset_is_empty2(base_model_name_dropdown,Huggingface_hub_token):
        if len(base_model_name_dropdown.strip()) == 0:
            raise gr.Error("Name is empty!")
        try:
            login_huggingface(Huggingface_hub_token,base_model_name_dropdown)
        except Exception as e:
            raise gr.Error(e)
    def check_local_model_or_dataset_is_empty3(base_model_name_dropdown,Huggingface_hub_token):
        if len(base_model_name_dropdown.strip()) == 0:
            raise gr.Error("Name is empty!")
        try:
            login_huggingface(Huggingface_hub_token,base_model_name_dropdown)
        except Exception as e:
            raise gr.Error(e)
    def check_local_model_or_dataset_is_empty4(base_model_name_dropdown,Huggingface_hub_token):
        if len(base_model_name_dropdown.strip()) == 0:
            raise gr.Error("Name is empty!")
        try:
            login_huggingface(Huggingface_hub_token,base_model_name_dropdown)
        except Exception as e:
            raise gr.Error(e)
    def check_local_model_or_dataset_is_empty5(base_model_name_dropdown,Huggingface_hub_token):
        if len(base_model_name_dropdown.strip()) == 0:
            raise gr.Error("Name is empty!")
        try:
            login_huggingface(Huggingface_hub_token,base_model_name_dropdown)
        except Exception as e:
            raise gr.Error(e)

    def download_hub_home_chat_model_postprocess():
        return gr.update(visible=True), gr.update(visible=False)


    def click_download_hub_home_chat_model_btn():
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)


    def click_stop_download_hub_home_chat_model_names_btn():
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)


    def click_stop_download_hub_home_chat_model_names_btn():
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)


    def change_home_chat_model_source_radio(home_chat_model_source_radio, hub_home_chat_model_names_dropdown):
        local_home_chat_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        if home_chat_model_source_radio == "Download From Huggingface Hub":
            if not hub_home_chat_model_names_dropdown:
                model_download_status = '<span style="color:red">&nbsp;&nbsp;&nbsp;&nbsp;No model is selected.</span>'
            else:
                if validate_model_path(hub_home_chat_model_names_dropdown)[0]:
                    model_download_status = '<span style="color:green">&nbsp;&nbsp;&nbsp;&nbsp;This model has already been downloaded to local,click load model to run.</span>'
                else:
                    model_download_status = '<span style="color:red">&nbsp;&nbsp;&nbsp;&nbsp;This model has not been downloaded.</span>'
            return gr.update(visible=True), gr.update(visible=False), gr.update(
                visible=False), gr.update(visible=True, value=model_download_status), gr.update(
                visible=True), gr.update(
                visible=False)
        else:
            model_download_status = ""
            return gr.update(visible=False), gr.update(visible=True), gr.update(
                visible=True), gr.update(visible=False, value=model_download_status), gr.update(
                visible=False), gr.update(
                visible=False)


    click_download_hub_home_chat_model_names_btn_event = download_hub_home_chat_model_names_btn.click(
        check_local_model_or_dataset_is_empty1, [hub_home_chat_model_names_dropdown,Huggingface_hub_token]).success(
        click_download_hub_home_chat_model_btn, [],
        [download_hub_home_chat_model_names_btn,
         stop_download_hub_home_chat_model_names_btn,
         download_hub_home_chat_model_status_markdown]).then(
        download_model_wrapper, [hub_home_chat_model_names_dropdown, local_home_chat_model_root_dir_textbox],
        download_hub_home_chat_model_status_markdown). \
        then(download_hub_home_chat_model_postprocess, [],
             [download_hub_home_chat_model_names_btn, stop_download_hub_home_chat_model_names_btn])

    stop_download_hub_home_chat_model_names_btn.click(click_stop_download_hub_home_chat_model_names_btn, [],
                                                      [download_hub_home_chat_model_names_btn,
                                                       stop_download_hub_home_chat_model_names_btn,
                                                       download_hub_home_chat_model_status_markdown],
                                                      cancels=[
                                                          click_download_hub_home_chat_model_names_btn_event])
    home_chat_model_source_radio.change(change_home_chat_model_source_radio,
                                        [home_chat_model_source_radio, hub_home_chat_model_names_dropdown],
                                        [hub_home_chat_model_names_dropdown, local_home_chat_model_names_dropdown,
                                         refresh_local_home_chat_model_names_btn,
                                         download_hub_home_chat_model_status_markdown,
                                         download_hub_home_chat_model_names_btn,
                                         stop_download_hub_home_chat_model_names_btn],
                                        cancels=[click_download_hub_home_chat_model_names_btn_event])
    def change_refresh_local_home_chat_model_names_btn():
        local_home_chat_model_names = get_hg_model_names_and_gguf_from_dir(local_home_chat_model_dir,runs_model_root_dir)
        return gr.update(choices=local_home_chat_model_names,value = local_home_chat_model_names[0] if local_home_chat_model_names else None)
    refresh_local_home_chat_model_names_btn.click(change_refresh_local_home_chat_model_names_btn,[],[local_home_chat_model_names_dropdown])
    def change_hub_home_chat_model_names_dropdown(hub_home_chat_model_names_dropdown):
        if not hub_home_chat_model_names_dropdown:
            return gr.update(visible=True,
                             value='<span style="color:red">&nbsp;&nbsp;&nbsp;&nbsp;No model is selected.</span>'), \
                   gr.update(visible=True), gr.update(visible=False)
        if validate_model_path(hub_home_chat_model_names_dropdown)[0]:
            return gr.update(
                visible=True,
                value='<span style="color:green">&nbsp;&nbsp;&nbsp;&nbsp;This model has already been downloaded to local,click load model to run.</span>'), \
                   gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=True,
                             value='<span style="color:red">&nbsp;&nbsp;&nbsp;&nbsp;This model has not been downloaded.</span>'), \
                   gr.update(visible=True), gr.update(visible=False)


    hub_home_chat_model_names_dropdown.change(change_hub_home_chat_model_names_dropdown,
                                              hub_home_chat_model_names_dropdown,
                                              [download_hub_home_chat_model_status_markdown,
                                               download_hub_home_chat_model_names_btn,
                                               stop_download_hub_home_chat_model_names_btn],
                                              cancels=[click_download_hub_home_chat_model_names_btn_event])


    def click_load_home_chat_model_btn(home_chat_model_source_radio, hub_home_chat_model_names_dropdown,
                                       local_home_chat_model_names_dropdown, max_new_tokens_slider,
                                       temperature_slider, top_k_slider, top_p_slider, repeat_penalty_slider,
                                       chat_history_window_slider,using_4bit_quantization_checkbox,low_cpu_mem_usage_checkbox,
                                       progress=gr.Progress()):

        if home_chat_model_source_radio == "Download From Huggingface Hub":
            cur_model_name = hub_home_chat_model_names_dropdown
        else:
            cur_model_name = local_home_chat_model_names_dropdown
        if not validate_model_path(cur_model_name)[0]:
            raise gr.Error(f"Model does not exist!")
        global infer_model
        global stop_generation_status
        stop_generation_status = True
        progress(0.6)
        if infer_model:
            infer_model.free_memory()
            infer_model = None
        torch.cuda.empty_cache()
        yield "Loading model ..."
        load_model_status = 0
        model_path = validate_model_path(cur_model_name)[1]
        if model_path.split('.')[-1] == "gguf":
            infer_model = LlamaCppInference(model_path=model_path, max_new_tokens=max_new_tokens_slider,
                                            temperature=temperature_slider,
                                            top_k=top_k_slider, top_p=top_p_slider,
                                            repetition_penalty=repeat_penalty_slider)
            load_model_status, msg = infer_model.load_model()
        else:
            infer_model = HuggingfaceInference(model_path=model_path, max_new_tokens=max_new_tokens_slider,
                                            temperature=temperature_slider,
                                            top_k=top_k_slider, top_p=top_p_slider,
                                            repetition_penalty=repeat_penalty_slider,
                                               using_4bit_quantization=using_4bit_quantization_checkbox,
                                               low_cpu_mem_usage=low_cpu_mem_usage_checkbox)
            load_model_status, msg = infer_model.load_model()
        if load_model_status == -1:
            raise gr.Error(f"Loading model error:{msg}")
            if infer_model:
                infer_model.free_memory()
                infer_model = None
            torch.cuda.empty_cache()
            return
        progress(1.0)
        return gr.update()

    def update_model_running_status():
        global chatbot_history
        return gr.update(visible=True,
                         value=f"<span style='color:green'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Model is runing ...</span>"),chatbot_history,gr.update()


    def show_model_running_status():
        return gr.update(visible=True)


    load_home_chat_model_btn.click(show_model_running_status, [], download_hub_home_chat_model_status_markdown).then(
        click_load_home_chat_model_btn, [home_chat_model_source_radio, hub_home_chat_model_names_dropdown,
                                       local_home_chat_model_names_dropdown, max_new_tokens_slider,
                                       temperature_slider, top_k_slider, top_p_slider, repeat_penalty_slider,
                                       chat_history_window_slider,using_4bit_quantization_checkbox,low_cpu_mem_usage_checkbox],
        [download_hub_home_chat_model_status_markdown]). \
        success(update_model_running_status, [], [download_hub_home_chat_model_status_markdown,chatbot,input_txtbox])


    def click_stop_btn():
        global stop_generation_status
        stop_generation_status = True


    def clear_chat_history():
        global chatbot_history, stop_generation_status
        stop_generation_status = True
        chatbot_history = []
        return gr.update(value=None)


    def show_chatbot_question1(text):
        global chatbot_history
        if not text:
            raise gr.Error('Enter text')
        chatbot_history = chatbot_history + [[text, '']]
        chatbot_history = chatbot_history[-5:]
        return chatbot_history
    def show_chatbot_question2(text):
        global chatbot_history
        if not text:
            raise gr.Error('Enter text')
        chatbot_history = chatbot_history + [[text, '']]
        chatbot_history = chatbot_history[-5:]
        return chatbot_history

    # input_txtbox.submit(add_text)
    def generate_btn_click1(input_txtbox):
        global chatbot_history, infer_model, stop_generation_status
        chatbot_history_np = np.asarray(chatbot_history)
        chatbot_history_np = chatbot_history_np.flatten()
        chatbot_history_list = chatbot_history_np.tolist()
        stop_generation_status = False
        model_type = "other model"
        if infer_model:
            model_type = get_model_type(infer_model.model_path)
            prompt = get_chat_history_prompt(chatbot_history_list, model_type)
            print(f"{model_type} input prompt:", prompt)
            answer = infer_model(prompt)
        else:
            raise gr.Error("Model is not loaded!")
            return chatbot_history,gr.update(value="")
        print(f"{model_type} output:", answer)
        for char in answer:
            if stop_generation_status:
                break
            try:
                chatbot_history[-1][-1] += char
            except:
                break
            time.sleep(0.05)
            # print("d2:",chatbot_history)
            yield chatbot_history,gr.update(value="")
        yield chatbot_history,gr.update(value="")
    def generate_btn_click2(input_txtbox):
        global chatbot_history, infer_model, stop_generation_status
        chatbot_history_np = np.asarray(chatbot_history)
        chatbot_history_np = chatbot_history_np.flatten()
        chatbot_history_list = chatbot_history_np.tolist()
        stop_generation_status = False
        running_model_name = "other model"
        if infer_model:
            if infer_model.model_path.lower().find("mistral") >= 0 and infer_model.model_path.lower().find(
                    "instruct") >= 0:
                running_model_name = "mistral"
                prompt = get_chat_history_prompt(chatbot_history_list, running_model_name)
            elif infer_model.model_path.lower().find("llama") >= 0 and infer_model.model_path.lower().find("chat") >= 0:
                running_model_name = "llama2"
                prompt = get_chat_history_prompt(chatbot_history_list, running_model_name)
            elif infer_model.model_path.lower().find("zephyr") >= 0:
                running_model_name = "zephyr"
                prompt = get_chat_history_prompt(chatbot_history_list, running_model_name)
            else:
                prompt = ','.join(chatbot_history_list[:-2])
                prompt = prompt + chatbot_history_list[-2]
            print(f"{running_model_name} input prompt:", prompt)
            answer = infer_model(prompt)
        else:
            raise gr.Error("Model is not loaded!")
            return chatbot_history,gr.update(value="")
        print(f"{running_model_name} output:", answer)
        for char in answer:
            if stop_generation_status:
                break
            try:
                chatbot_history[-1][-1] += char
            except:
                break
            time.sleep(0.05)
            # print("d2:",chatbot_history)
            yield chatbot_history,gr.update(value="")
        yield chatbot_history,gr.update(value="")

    def generate_btn_click_clear_text1():
        return gr.update(value="")
    def generate_btn_click_clear_text2():
        return gr.update(value="")
    input_txtbox.submit(show_chatbot_question1, inputs=[input_txtbox], outputs=[chatbot], queue=False). \
        success(generate_btn_click1, inputs=[input_txtbox], outputs=[chatbot,input_txtbox])
    generate_btn.click(show_chatbot_question2, inputs=[input_txtbox], outputs=[chatbot], queue=False). \
        success(generate_btn_click2, inputs=[input_txtbox], outputs=[chatbot,input_txtbox])
    # clear_btn.click(clear_chat_history, [], chatbot)
    stop_btn.click(click_stop_btn)

    ##########################
    ######################

    def click_delete_text_btn(training_runs_dropdown):

            delete_run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', training_runs_dropdown)
            if os.path.exists(delete_run_dir):
                shutil.rmtree(delete_run_dir)

            training_runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')
            run_names = os.listdir(training_runs_dir)
            run_names = [run_name for run_name in run_names if
                         os.path.isdir(os.path.join(training_runs_dir, run_name))]
            run_names.sort(key=lambda f: os.path.getmtime(os.path.join(training_runs_dir, f)))
            sss = np.random.randint(0, 100) + 1000
            iframe = f'<iframe src={TENSORBOARD_URL} style="border:none;height:{sss}px;width:100%">'

            return gr.update(choices=run_names, value=run_names[0] if run_names else ""), gr.update(value=iframe)

    def click_stop_training_btn():
        global stop_training
        if TRAINING_STATUS.status == 0:
            TRAINING_STATUS.status = 1
            gr.Warning('Training is stopping!')
        elif TRAINING_STATUS.status == 1:
            gr.Warning('Training is stopping!')
        else:
            gr.Warning('Training has already been stopped!')


    stop_training_btn.click(click_stop_training_btn, [])


    def select_tensorboard_tab():
        # ping_tensorboard_cmd = "nc -zv localhost 6006"
        try:
            # ping_output = subprocess.check_output(ping_tensorboard_cmd, shell=True)
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((LOCAL_HOST_IP, 6006))
            s.close()
        except Exception as e:
            return gr.update(value="The tensorboard could be temporarily unavailable. Try switch back in a few moments.")
        sss = np.random.randint(0, 100) + 1000
        iframe = f'<iframe src={TENSORBOARD_URL} style="border:none;height:{sss}px;width:100%">'
        return gr.update(value=iframe)

    tensorboard_tab.select(select_tensorboard_tab, [], tensorboard_html)


    def click_download_local_dataset_btn():
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)


    def click_stop_download_local_dataset_btn():
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)


    def click_refresh_local_train_dataset_list_btn():
        local_dataset_list, _ = get_local_dataset_list()
        return gr.update(choices=local_dataset_list,
                         value=local_dataset_list[0] if len(local_dataset_list) > 0 else None)
        # return  gr.update(choices=local_dataset_list+[" "],value=" ")


    def get_hg_dataset_train_val_set(hg_dataset_path_textbox, local_dataset_root_dir_textbox):

        hg_dataset_dir = os.path.join(local_dataset_root_dir_textbox, hg_dataset_path_textbox)
        dataset_config_path1 = os.path.join(hg_dataset_dir, "dataset_dict.json")
        dataset_config_path2 = os.path.join(hg_dataset_dir, "dataset_infos.json")
        if not os.path.exists(dataset_config_path1) and not os.path.exists(
                dataset_config_path2):
            raise gr.Warning(f"Invalid HG Dataset:{hg_dataset_dir}")

            return gr.update(), gr.update(), gr.update(), gr.update()
        else:
            DATASET_FIRST_ROW, split_list = get_first_row_from_dataset(hg_dataset_dir)
            if "train" in split_list:
                split_list.pop(split_list.index("train"))
                split_list.insert(0, "train")

            return gr.update(choices=split_list, value=split_list[0] if split_list else None), gr.update(
                choices=split_list, value=None), gr.update(visible=True), gr.update(visible=False)


    def change_hg_dataset_path_textbox(hg_dataset_path_textbox):
        global local_dataset_root_dir
        hg_dataset_path_textbox = hg_dataset_path_textbox.strip()
        if hg_dataset_path_textbox == "":
            return gr.update(
                value="<span style='color:red'>&nbsp;&nbsp;&nbsp;&nbsp;This Dataset's name is empty!</span>"), gr.update(
                choices=[], value=None), gr.update(choices=[], value=None)
        else:

            hg_dataset_dir = os.path.join(local_dataset_root_dir,
                                          hg_dataset_path_textbox)
            dataset_config_path1 = os.path.join(hg_dataset_dir, "dataset_dict.json")
            dataset_config_path2 = os.path.join(hg_dataset_dir, "dataset_infos.json")
            if not os.path.exists(dataset_config_path1) and not os.path.exists(
                    dataset_config_path2):
                # raise gr.Warning(f"Invalid HG Dataset:{hg_dataset_dir}")
                return gr.update(
                    value="<span style='color:red'>&nbsp;&nbsp;&nbsp;&nbsp;This Dataset has not been downloaded.</span>"), gr.update(
                    choices=[], value=None), gr.update(choices=[], value=None)
            else:
                DATASET_FIRST_ROW, split_list = get_first_row_from_dataset(hg_dataset_dir)
                if "train" in split_list:
                    split_list.pop(split_list.index("train"))
                    split_list.insert(0, "train")
                return gr.update(
                    value="<span style='color:green'>&nbsp;&nbsp;&nbsp;&nbsp;This Dataset has already been downloaded.</span>"), gr.update(
                    choices=split_list, value=split_list[0] if split_list else None), gr.update(choices=split_list,
                                                                                                value="")


    hg_dataset_path_textbox.change(change_hg_dataset_path_textbox, hg_dataset_path_textbox,
                                   [download_dataset_status_markdown, hg_train_dataset_dropdown,
                                    hg_val_dataset_dropdown])

    def upload_val_file(chat_with_file_btn):
        doc_path = chat_with_file_btn.name
        return gr.update(value=doc_path),gr.update(visible=True)


    def change_dataset_source(dataset_source_radio,base_model_source_radio,base_model_name_dropdown,local_model_dropdown,hg_dataset_path_textbox):
        global DATASET_FIRST_ROW

        if dataset_source_radio == "Download From Huggingface Hub":
            hg_dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets',
                                           hg_dataset_path_textbox)
            dataset_config_path1 = os.path.join(hg_dataset_dir, "dataset_dict.json")
            dataset_config_path2 = os.path.join(hg_dataset_dir, "dataset_infos.json")
            if not os.path.exists(dataset_config_path1) and not os.path.exists(dataset_config_path2):
                # gr.Warning(f"Invalid HG Dataset:{hg_dataset_dir}")
                return gr.update(), gr.update(), gr.update(), gr.update(visible=True), gr.update(
                    visible=True), gr.update(visible=False), \
                       gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
                    visible=True), \
                       gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), \
                       gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(
                    visible=True), \
                       gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(
                    visible=True)
            else:
                DATASET_FIRST_ROW, split_list = get_first_row_from_dataset(hg_dataset_dir)
                if "train" in split_list:
                    split_list.pop(split_list.index("train"))
                    split_list.insert(0,"train")
                col_names = list(DATASET_FIRST_ROW)

                new_col_names = []
                for col_name in col_names:
                    if col_name.lower() != "id":
                        new_col_names.append(col_name)
                INIT_COL1_TEXT = ""
                if len(new_col_names) > 0:
                    INIT_COL1_TEXT = new_col_names[0]
                INIT_COL2_TEXT = ""
                if len(new_col_names) > 1:
                    INIT_COL2_TEXT = new_col_names[1]
                INIT_COL3_TEXT = ""
                if len(new_col_names) > 2:
                    INIT_COL3_TEXT = new_col_names[2]
                INIT_COL4_TEXT = ""
                if len(new_col_names) > 3:
                    INIT_COL4_TEXT = new_col_names[3]
                col_names.insert(0, "")
                if base_model_source_radio == "Download From Huggingface Hub":
                    curr_model_name = base_model_name_dropdown
                else:
                    curr_model_name = local_model_dropdown

                model_type = get_model_type(curr_model_name)
                if not curr_model_name or model_type == "other model":
                    return gr.update(), \
                           gr.update(choices=split_list, value=split_list[0] if split_list else None), \
                           gr.update(choices=split_list, value=None), \
                           gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), \
                           gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), \
                           gr.update(visible=False), gr.update(visible=True), gr.update(visible=True),\
                           gr.update(value="", visible=True), gr.update(choices=col_names, value=INIT_COL1_TEXT), \
                           gr.update(value="", visible=True), gr.update(choices=col_names, value=INIT_COL2_TEXT), \
                           gr.update(value="", visible=True), gr.update(choices=col_names, value=INIT_COL3_TEXT), \
                           gr.update(value="", visible=True), gr.update(choices=col_names, value=INIT_COL4_TEXT)
                else:
                    if "instruction" in col_names and "input" in col_names and "output" in col_names:
                        if model_type == "mistral" or model_type == "llama2":
                            INIT_PREFIX1 = "<s>[INST] "
                            INIT_PREFIX2 = "here are the inputs "
                            INIT_PREFIX3 = " [/INST]"
                            INIT_PREFIX4 = "</s>"
                        else:
                            INIT_PREFIX1 = "<|user|>\n"
                            INIT_PREFIX2 = "here are the inputs "
                            INIT_PREFIX3 = "</s><|assistant|>\n"
                            INIT_PREFIX4 = "</s>"
                        return gr.update(), \
                               gr.update(choices=split_list, value=split_list[0] if split_list else None), \
                               gr.update(choices=split_list, value=None), \
                               gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), \
                               gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
                            visible=True), \
                               gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), \
                               gr.update(value=INIT_PREFIX1, visible=True), gr.update(choices=col_names,
                                                                                      value=col_names[col_names.index(
                                                                                          "instruction")]), \
                               gr.update(value=INIT_PREFIX2, visible=True), gr.update(choices=col_names,
                                                                                      value=col_names[
                                                                                          col_names.index("input")]), \
                               gr.update(value=INIT_PREFIX3, visible=True), gr.update(choices=col_names,
                                                                                      value=col_names[
                                                                                          col_names.index("output")]), \
                               gr.update(value=INIT_PREFIX4, visible=True), gr.update(choices=col_names, value="")
                    else:
                        if model_type == "mistral" or model_type == "llama2":
                            INIT_PREFIX1 = "<s>[INST] "
                            INIT_PREFIX2 = " [/INST]"
                            INIT_PREFIX3 = "</s>"
                            INIT_PREFIX4 = ""
                        else:
                            INIT_PREFIX1 = "<|user|>\n"
                            INIT_PREFIX2 = "</s><|assistant|>\n"
                            INIT_PREFIX3 = "</s>"
                            INIT_PREFIX4 = ""


                        return gr.update(), \
                               gr.update(choices=split_list, value=split_list[0] if split_list else None), \
                               gr.update(choices=split_list, value=None), \
                               gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), \
                               gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), \
                               gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), \
                               gr.update(visible=True), \
                               gr.update(value=INIT_PREFIX1, visible=True), gr.update(choices=col_names,value=INIT_COL1_TEXT), \
                               gr.update(value=INIT_PREFIX2, visible=True), gr.update(choices=col_names,
                                                                                      value=INIT_COL2_TEXT), \
                               gr.update(value=INIT_PREFIX3, visible=True), gr.update(choices=col_names,
                                                                                      value=INIT_COL3_TEXT), \
                               gr.update(value=INIT_PREFIX4, visible=True), gr.update(choices=col_names,
                                                                                      value=INIT_COL4_TEXT)
        else:
            local_dataset_list, _ = get_local_dataset_list()
            if local_dataset_list:
                local_dataset_list.pop(local_dataset_list.index(INIT_DATASET_NAME))
                local_dataset_list.insert(0,INIT_DATASET_NAME)
                local_dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets',
                                               local_dataset_list[0])


                DATASET_FIRST_ROW, split_list = get_first_row_from_dataset(local_dataset_path)
                if "train" in split_list:
                    split_list.pop(split_list.index("train"))
                    split_list.insert(0,"train")
            else:
                split_list = []

            col_names = list(DATASET_FIRST_ROW)
            new_col_names = []
            for col_name in col_names:
                if col_name.lower() != "id":
                    new_col_names.append(col_name)
            INIT_COL1_TEXT = ""
            if len(new_col_names) > 0:
                INIT_COL1_TEXT = new_col_names[0]
            INIT_COL2_TEXT = ""
            if len(new_col_names) > 1:
                INIT_COL2_TEXT = new_col_names[1]
            INIT_COL3_TEXT = ""
            if len(new_col_names) > 2:
                INIT_COL3_TEXT = new_col_names[2]
            INIT_COL4_TEXT = ""
            if len(new_col_names) > 3:
                INIT_COL4_TEXT = new_col_names[3]
            col_names.insert(0, "")
            if base_model_source_radio == "Download From Huggingface Hub":
                curr_model_name = base_model_name_dropdown
            else:
                curr_model_name = local_model_dropdown
            model_type = get_model_type(curr_model_name)
            if not curr_model_name or model_type == "other model":
                return gr.update(choices=local_dataset_list,value=local_dataset_list[0] if local_dataset_list else None),\
                       gr.update(choices=split_list, value=split_list[0] if split_list else None),\
                       gr.update(choices=split_list, value=None), \
                       gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), \
                       gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), \
                       gr.update(visible=False), gr.update(visible=False), \
                       gr.update(value="",visible=True), gr.update(choices=col_names, value=INIT_COL1_TEXT),\
                       gr.update(value="",visible=True), gr.update(choices=col_names,value=INIT_COL2_TEXT),  \
                       gr.update(value="",visible=True),gr.update(choices=col_names, value=INIT_COL3_TEXT), \
                       gr.update(value="",visible=True),gr.update(choices=col_names, value=INIT_COL4_TEXT)
            else:
                if "instruction" in col_names and "input" in col_names and "output" in col_names:
                    if model_type == "mistral" or model_type == "llama2":
                        INIT_PREFIX1 = "<s>[INST] "
                        INIT_PREFIX2 = "here are the inputs "
                        INIT_PREFIX3 = " [/INST]"
                        INIT_PREFIX4 = "</s>"
                    else:
                        INIT_PREFIX1 = "<|user|>\n"
                        INIT_PREFIX2 = "here are the inputs "
                        INIT_PREFIX3 = "</s><|assistant|>\n"
                        INIT_PREFIX4 = "</s>"

                    return gr.update(choices=local_dataset_list,value=local_dataset_list[0] if local_dataset_list else None),\
                           gr.update(choices=split_list, value=split_list[0] if split_list else None), \
                           gr.update(choices=split_list, value=None),\
                           gr.update(visible=False),gr.update(visible=False), gr.update(visible=True),  gr.update(visible=True), \
                           gr.update(visible=True),gr.update(visible=True),gr.update(visible=False),gr.update(visible=False),\
                           gr.update(visible=False),gr.update(visible=False),\
                           gr.update(value=INIT_PREFIX1,visible=True), gr.update(choices=col_names, value=col_names[col_names.index("instruction")]), \
                           gr.update(value=INIT_PREFIX2,visible=True), gr.update(choices=col_names, value=col_names[col_names.index("input")]), \
                           gr.update(value=INIT_PREFIX3,visible=True), gr.update(choices=col_names,value=col_names[col_names.index("output")]), \
                           gr.update(value=INIT_PREFIX4,visible=True), gr.update(choices=col_names, value="")
                else:
                    if model_type == "mistral" or model_type == "llama2":
                        INIT_PREFIX1 = "<s>[INST] "
                        INIT_PREFIX2 = " [/INST]"
                        INIT_PREFIX3 = "</s>"
                        INIT_PREFIX4 = ""
                    else:
                        INIT_PREFIX1 = "<|user|>\n"
                        INIT_PREFIX2 = "</s><|assistant|>\n"
                        INIT_PREFIX3 = "</s>"
                        INIT_PREFIX4 = ""

                    return gr.update(choices=local_dataset_list,value=local_dataset_list[0] if local_dataset_list else None),\
                           gr.update(choices=split_list, value=split_list[0] if split_list else None), \
                           gr.update(choices=split_list, value=None),\
                           gr.update(visible=False),gr.update(visible=False), gr.update(visible=True),  gr.update(visible=True), \
                           gr.update(visible=True),gr.update(visible=True),gr.update(visible=False),gr.update(visible=False),\
                           gr.update(visible=False),gr.update(visible=False),\
                           gr.update(value=INIT_PREFIX1,visible=True), gr.update(choices=col_names, value=INIT_COL1_TEXT), \
                           gr.update(value=INIT_PREFIX2,visible=True), gr.update(choices=col_names, value=INIT_COL2_TEXT),\
                           gr.update(value=INIT_PREFIX3,visible=True), gr.update(choices=col_names, value=INIT_COL3_TEXT), \
                           gr.update(value=INIT_PREFIX4,visible=True), gr.update(choices=col_names, value=INIT_COL4_TEXT)

    def change_dataset_col1(prefix1_textbox,datatset_col1_dropdown,prefix2_textbox,datatset_col2_dropdown,prefix3_textbox,datatset_col3_dropdown,prefix4_textbox,datatset_col4_dropdown):
        prompt = prefix1_textbox
        if  isinstance(datatset_col1_dropdown,str) and datatset_col1_dropdown in DATASET_FIRST_ROW:
            prompt += str(DATASET_FIRST_ROW[datatset_col1_dropdown])
            prompt += prefix2_textbox
        if  isinstance(datatset_col2_dropdown,str) and datatset_col2_dropdown in DATASET_FIRST_ROW:
            prompt += str(DATASET_FIRST_ROW[datatset_col2_dropdown])
            prompt += prefix3_textbox
        if  isinstance(datatset_col3_dropdown,str) and datatset_col3_dropdown in DATASET_FIRST_ROW:
            prompt += str(DATASET_FIRST_ROW[datatset_col3_dropdown])
            prompt += prefix4_textbox
        if  isinstance(datatset_col4_dropdown,str) and datatset_col4_dropdown in DATASET_FIRST_ROW:
            prompt += str(DATASET_FIRST_ROW[datatset_col4_dropdown])
        return prompt
    def change_dataset_col2(prefix1_textbox,datatset_col1_dropdown,prefix2_textbox,datatset_col2_dropdown,prefix3_textbox,datatset_col3_dropdown,prefix4_textbox,datatset_col4_dropdown):
        prompt = prefix1_textbox
        if  isinstance(datatset_col1_dropdown,str) and datatset_col1_dropdown in DATASET_FIRST_ROW:
            prompt += str(DATASET_FIRST_ROW[datatset_col1_dropdown])
            prompt += prefix2_textbox
        if  isinstance(datatset_col2_dropdown,str) and datatset_col2_dropdown in DATASET_FIRST_ROW:
            prompt += str(DATASET_FIRST_ROW[datatset_col2_dropdown])
            prompt += prefix3_textbox
        if  isinstance(datatset_col3_dropdown,str) and datatset_col3_dropdown in DATASET_FIRST_ROW:
            prompt += str(DATASET_FIRST_ROW[datatset_col3_dropdown])
            prompt += prefix4_textbox
        if  isinstance(datatset_col4_dropdown,str) and datatset_col4_dropdown in DATASET_FIRST_ROW:
            prompt += str(DATASET_FIRST_ROW[datatset_col4_dropdown])
        return prompt
    def change_dataset_col3(prefix1_textbox,datatset_col1_dropdown,prefix2_textbox,datatset_col2_dropdown,prefix3_textbox,datatset_col3_dropdown,prefix4_textbox,datatset_col4_dropdown):
        prompt = prefix1_textbox
        if  isinstance(datatset_col1_dropdown,str) and datatset_col1_dropdown in DATASET_FIRST_ROW:
            prompt += str(DATASET_FIRST_ROW[datatset_col1_dropdown])
            prompt += prefix2_textbox
        if  isinstance(datatset_col2_dropdown,str) and datatset_col2_dropdown in DATASET_FIRST_ROW:
            prompt += str(DATASET_FIRST_ROW[datatset_col2_dropdown])
            prompt += prefix3_textbox
        if  isinstance(datatset_col3_dropdown,str) and datatset_col3_dropdown in DATASET_FIRST_ROW:
            prompt += str(DATASET_FIRST_ROW[datatset_col3_dropdown])
            prompt += prefix4_textbox
        if  isinstance(datatset_col4_dropdown,str) and datatset_col4_dropdown in DATASET_FIRST_ROW:
            prompt += str(DATASET_FIRST_ROW[datatset_col4_dropdown])
        return prompt
    def change_dataset_col4(prefix1_textbox,datatset_col1_dropdown,prefix2_textbox,datatset_col2_dropdown,prefix3_textbox,datatset_col3_dropdown,prefix4_textbox,datatset_col4_dropdown):
        prompt = prefix1_textbox
        if  isinstance(datatset_col1_dropdown,str) and datatset_col1_dropdown in DATASET_FIRST_ROW:
            prompt += str(DATASET_FIRST_ROW[datatset_col1_dropdown])
            prompt += prefix2_textbox
        if  isinstance(datatset_col2_dropdown,str) and datatset_col2_dropdown in DATASET_FIRST_ROW:
            prompt += str(DATASET_FIRST_ROW[datatset_col2_dropdown])
            prompt += prefix3_textbox
        if  isinstance(datatset_col3_dropdown,str) and datatset_col3_dropdown in DATASET_FIRST_ROW:
            prompt += str(DATASET_FIRST_ROW[datatset_col3_dropdown])
            prompt += prefix4_textbox
        if  isinstance(datatset_col4_dropdown,str) and datatset_col4_dropdown in DATASET_FIRST_ROW:
            prompt += str(DATASET_FIRST_ROW[datatset_col4_dropdown])
        return prompt
    def change_prefix1_textbox(prefix1_textbox,datatset_col1_dropdown,prefix2_textbox,datatset_col2_dropdown,prefix3_textbox,datatset_col3_dropdown,prefix4_textbox,datatset_col4_dropdown):
        prompt = prefix1_textbox
        if  isinstance(datatset_col1_dropdown,str) and datatset_col1_dropdown in DATASET_FIRST_ROW:
            prompt += str(DATASET_FIRST_ROW[datatset_col1_dropdown])
            prompt += prefix2_textbox
        if  isinstance(datatset_col2_dropdown,str) and datatset_col2_dropdown in DATASET_FIRST_ROW:
            prompt += str(DATASET_FIRST_ROW[datatset_col2_dropdown])
            prompt += prefix3_textbox
        if  isinstance(datatset_col3_dropdown,str) and datatset_col3_dropdown in DATASET_FIRST_ROW:
            prompt += str(DATASET_FIRST_ROW[datatset_col3_dropdown])
            prompt += prefix4_textbox
        if  isinstance(datatset_col4_dropdown,str) and datatset_col4_dropdown in DATASET_FIRST_ROW:
            prompt += str(DATASET_FIRST_ROW[datatset_col4_dropdown])
        return prompt
    def change_prefix2_textbox(prefix1_textbox,datatset_col1_dropdown,prefix2_textbox,datatset_col2_dropdown,prefix3_textbox,datatset_col3_dropdown,prefix4_textbox,datatset_col4_dropdown):
        prompt = prefix1_textbox
        if  isinstance(datatset_col1_dropdown,str) and datatset_col1_dropdown in DATASET_FIRST_ROW:
            prompt += str(DATASET_FIRST_ROW[datatset_col1_dropdown])
            prompt += prefix2_textbox
        if  isinstance(datatset_col2_dropdown,str) and datatset_col2_dropdown in DATASET_FIRST_ROW:
            prompt += str(DATASET_FIRST_ROW[datatset_col2_dropdown])
            prompt += prefix3_textbox
        if  isinstance(datatset_col3_dropdown,str) and datatset_col3_dropdown in DATASET_FIRST_ROW:
            prompt += str(DATASET_FIRST_ROW[datatset_col3_dropdown])
            prompt += prefix4_textbox
        if  isinstance(datatset_col4_dropdown,str) and datatset_col4_dropdown in DATASET_FIRST_ROW:
            prompt += str(DATASET_FIRST_ROW[datatset_col4_dropdown])
        return prompt
    def change_prefix3_textbox(prefix1_textbox,datatset_col1_dropdown,prefix2_textbox,datatset_col2_dropdown,prefix3_textbox,datatset_col3_dropdown,prefix4_textbox,datatset_col4_dropdown):
        prompt = prefix1_textbox
        if  isinstance(datatset_col1_dropdown,str) and datatset_col1_dropdown in DATASET_FIRST_ROW:
            prompt += str(DATASET_FIRST_ROW[datatset_col1_dropdown])
            prompt += prefix2_textbox
        if  isinstance(datatset_col2_dropdown,str) and datatset_col2_dropdown in DATASET_FIRST_ROW:
            prompt += str(DATASET_FIRST_ROW[datatset_col2_dropdown])
            prompt += prefix3_textbox
        if  isinstance(datatset_col3_dropdown,str) and datatset_col3_dropdown in DATASET_FIRST_ROW:
            prompt += str(DATASET_FIRST_ROW[datatset_col3_dropdown])
            prompt += prefix4_textbox
        if  isinstance(datatset_col4_dropdown,str) and datatset_col4_dropdown in DATASET_FIRST_ROW:
            prompt += str(DATASET_FIRST_ROW[datatset_col4_dropdown])
        return prompt
    def change_prefix4_textbox(prefix1_textbox,datatset_col1_dropdown,prefix2_textbox,datatset_col2_dropdown,prefix3_textbox,datatset_col3_dropdown,prefix4_textbox,datatset_col4_dropdown):
        prompt = prefix1_textbox
        if  isinstance(datatset_col1_dropdown,str) and datatset_col1_dropdown in DATASET_FIRST_ROW:
            prompt += str(DATASET_FIRST_ROW[datatset_col1_dropdown])
            prompt += prefix2_textbox
        if  isinstance(datatset_col2_dropdown,str) and datatset_col2_dropdown in DATASET_FIRST_ROW:
            prompt += str(DATASET_FIRST_ROW[datatset_col2_dropdown])
            prompt += prefix3_textbox
        if  isinstance(datatset_col3_dropdown,str) and datatset_col3_dropdown in DATASET_FIRST_ROW:
            prompt += str(DATASET_FIRST_ROW[datatset_col3_dropdown])
            prompt += prefix4_textbox
        if  isinstance(datatset_col4_dropdown,str) and datatset_col4_dropdown in DATASET_FIRST_ROW:
            prompt += str(DATASET_FIRST_ROW[datatset_col4_dropdown])
        return prompt

    def change_base_model_source(base_model_source_radio,base_model_name_dropdown,local_model_dropdown,local_train_path_dataset_dropdown):
        if base_model_source_radio == "Download From Huggingface Hub":
            if not base_model_name_dropdown:
                model_download_status =  '<span style="color:red">&nbsp;&nbsp;&nbsp;&nbsp;No model is selected.</span>'
            else:
                if validate_model_path(base_model_name_dropdown)[0]:
                    model_download_status = '<span style="color:green">&nbsp;&nbsp;&nbsp;&nbsp;This model has already been downloaded to local.</span>'
                else:
                    model_download_status =  '<span style="color:red">&nbsp;&nbsp;&nbsp;&nbsp;This model has not been downloaded.</span>'

            return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(value=""), gr.update(value=model_download_status), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
        else:
            model_download_status = ""

            global DATASET_FIRST_ROW
            local_dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets',
                                              local_train_path_dataset_dropdown)
            DATASET_FIRST_ROW, split_list = get_first_row_from_dataset(local_dataset_path)
            if "train" in split_list:
                split_list.pop(split_list.index("train"))
                split_list.insert(0, "train")

            col_names = list(DATASET_FIRST_ROW)

            new_col_names = []
            for col_name in col_names:
                if col_name.lower() != "id":
                    new_col_names.append(col_name)
            INIT_COL1_TEXT = ""
            if len(new_col_names) > 0:
                INIT_COL1_TEXT = new_col_names[0]
            INIT_COL2_TEXT = ""
            if len(new_col_names) > 1:
                INIT_COL2_TEXT = new_col_names[1]
            INIT_COL3_TEXT = ""
            if len(new_col_names) > 2:
                INIT_COL3_TEXT = new_col_names[2]
            INIT_COL4_TEXT = ""
            if len(new_col_names) > 3:
                INIT_COL4_TEXT = new_col_names[3]
            col_names.insert(0, "")


            model_type = get_model_type(local_model_dropdown)

            if model_type == "other model":
                if "instruction" in col_names and "input" in col_names and "output" in col_names:
                    INIT_PREFIX1 = "<s>[INST] "
                    INIT_PREFIX2 = "here are the inputs "
                    INIT_PREFIX3 = " [/INST]"
                    INIT_PREFIX4 = "</s>"
                else:
                    INIT_PREFIX1 = "<s>[INST] "
                    INIT_PREFIX2 = " [/INST]"
                    INIT_PREFIX3 = "</s>"
                    INIT_PREFIX4 = ""
            else:
                if "instruction" in col_names and "input" in col_names and "output" in col_names:
                    if model_type == "mistral" or model_type == "llama2":
                        INIT_PREFIX1 = "<s>[INST] "
                        INIT_PREFIX2 = "here are the inputs "
                        INIT_PREFIX3 = " [/INST]"
                        INIT_PREFIX4 = "</s>"
                    else:
                        INIT_PREFIX1 = "<|user|>\n"
                        INIT_PREFIX2 = "here are the inputs "
                        INIT_PREFIX3 = "</s><|assistant|>\n"
                        INIT_PREFIX4 = "</s>"
                else:
                    if model_type == "mistral" or model_type == "llama2":
                        INIT_PREFIX1 = "<s>[INST] "
                        INIT_PREFIX2 = " [/INST]"
                        INIT_PREFIX3 = "</s>"
                        INIT_PREFIX4 = ""
                    else:
                        INIT_PREFIX1 = "<|user|>\n"
                        INIT_PREFIX2 = "</s><|assistant|>\n"
                        INIT_PREFIX3 = "</s>"
                        INIT_PREFIX4 = ""

            return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(value=""), gr.update(value=model_download_status), gr.update(visible=False), gr.update(visible=False),INIT_PREFIX1, INIT_PREFIX2, INIT_PREFIX3, INIT_PREFIX4
    def click_refresh_local_model_list_btn(refresh_local_model_list_btn):
        # local_model_list,_ = get_local_model_list()
        local_chat_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        runs_model_root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
        local_model_list = get_hg_model_names_and_gguf_from_dir(local_chat_model_dir, runs_model_root_dir)
        return gr.update(choices=local_model_list, value=local_model_list[0] if len(local_model_list) > 0 else None)
    def change_base_model_name_dropdown(base_model_name_dropdown):

        if not base_model_name_dropdown:
            return "","","","",gr.update(visible=True,value='<span style="color:red">&nbsp;&nbsp;&nbsp;&nbsp;No model is selected.</span>'), gr.update(visible=True), gr.update(visible=False)

        model_type = get_model_type(base_model_name_dropdown)
        if model_type == "mistral" or model_type == "llama2":
            INIT_PREFIX1 = "<s>[INST] "
            INIT_PREFIX2 = "here are the inputs "
            INIT_PREFIX3 = " [/INST]"
            INIT_PREFIX4 = "</s>"
        elif model_type == "zephyr":
            INIT_PREFIX1 = "<|user|>\n"
            INIT_PREFIX2 = "here are the inputs "
            INIT_PREFIX3 = "</s><|assistant|>\n"
            INIT_PREFIX4 = "</s>"
        else:
            INIT_PREFIX1 = ""
            INIT_PREFIX2 = ""
            INIT_PREFIX3 = ""
            INIT_PREFIX4 = ""

        if validate_model_path(base_model_name_dropdown)[0]:
            return INIT_PREFIX1,INIT_PREFIX2,INIT_PREFIX3,INIT_PREFIX4,gr.update(visible=True,value='<span style="color:green">&nbsp;&nbsp;&nbsp;&nbsp;This model has already been downloaded to local.</span>'), gr.update(visible=True), gr.update(visible=False)
        else:
            return INIT_PREFIX1,INIT_PREFIX2,INIT_PREFIX3,INIT_PREFIX4,gr.update(visible=True,value='<span style="color:red">&nbsp;&nbsp;&nbsp;&nbsp;This model has not been downloaded.</span>'), gr.update(visible=True), gr.update(visible=False)

    def change_local_train_path_dataset_dropdown(local_train_path_dataset_dropdown,base_model_name_dropdown,base_model_source_radio,local_model_dropdown):
        global DATASET_FIRST_ROW
        local_dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets',local_train_path_dataset_dropdown)
        DATASET_FIRST_ROW, split_list = get_first_row_from_dataset(local_dataset_path)
        if "train" in split_list:
            split_list.pop(split_list.index("train"))
            split_list.insert(0, "train")

        col_names = list(DATASET_FIRST_ROW)

        new_col_names = []
        for col_name in col_names:
            if col_name.lower() != "id":
                new_col_names.append(col_name)
        INIT_COL1_TEXT = ""
        if len(new_col_names) > 0:
            INIT_COL1_TEXT = new_col_names[0]
        INIT_COL2_TEXT = ""
        if len(new_col_names) > 1:
            INIT_COL2_TEXT = new_col_names[1]
        INIT_COL3_TEXT = ""
        if len(new_col_names) > 2:
            INIT_COL3_TEXT = new_col_names[2]
        INIT_COL4_TEXT = ""
        if len(new_col_names) > 3:
            INIT_COL4_TEXT = new_col_names[3]
        col_names.insert(0, "")

        if base_model_source_radio == "Download From Huggingface Hub":
            curr_model_name = base_model_name_dropdown
        else:
            curr_model_name = local_model_dropdown
        model_type = get_model_type(curr_model_name)
        if not curr_model_name or model_type == "other model":
            return gr.update(choices=split_list, value=split_list[0] if split_list else None),gr.update(choices=split_list, value=None),\
                   "", gr.update(choices=col_names, value=INIT_COL1_TEXT), "", gr.update(choices=col_names,
                                                                                         value=INIT_COL2_TEXT), "", gr.update(
                choices=col_names, value=INIT_COL3_TEXT), "", gr.update(choices=col_names, value=INIT_COL4_TEXT),
        else:

            if "instruction" in col_names and "input" in col_names and "output" in col_names:
                if model_type == "mistral" or model_type == "llama2":
                    INIT_PREFIX1 = "<s>[INST] "
                    INIT_PREFIX2 = "here are the inputs "
                    INIT_PREFIX3 = " [/INST]"
                    INIT_PREFIX4 = "</s>"
                else:
                    INIT_PREFIX1 = "<|user|>\n"
                    INIT_PREFIX2 = "here are the inputs "
                    INIT_PREFIX3 = "</s><|assistant|>\n"
                    INIT_PREFIX4 = "</s>"

                return gr.update(choices=split_list, value=split_list[0] if split_list else None),gr.update(choices=split_list, value=None),\
                       INIT_PREFIX1, gr.update(choices=col_names,value=col_names[col_names.index("instruction")]), INIT_PREFIX2, gr.update(choices=col_names, value=col_names[col_names.index("input")]), \
                       INIT_PREFIX3, gr.update(choices=col_names, value=col_names[col_names.index("output")]), INIT_PREFIX4, gr.update(choices=col_names, value="")
            else:
                if model_type == "mistral" or model_type == "llama2":
                    INIT_PREFIX1 = "<s>[INST] "
                    INIT_PREFIX2 = " [/INST]"
                    INIT_PREFIX3 = "</s>"
                    INIT_PREFIX4 = ""
                else:
                    INIT_PREFIX1 = "<|user|>\n"
                    INIT_PREFIX2 = "</s><|assistant|>\n"
                    INIT_PREFIX3 = "</s>"
                    INIT_PREFIX4 = ""

                return gr.update(choices=split_list, value=split_list[0] if split_list else None),gr.update(choices=split_list, value=None),\
                       INIT_PREFIX1, gr.update(choices=col_names, value=INIT_COL1_TEXT), INIT_PREFIX2, gr.update(choices=col_names, value=INIT_COL2_TEXT), \
                       INIT_PREFIX3, gr.update(choices=col_names,value=INIT_COL3_TEXT), INIT_PREFIX4, gr.update(choices=col_names, value=INIT_COL4_TEXT)

    local_train_path_dataset_dropdown.change(change_local_train_path_dataset_dropdown,[local_train_path_dataset_dropdown,base_model_name_dropdown,base_model_source_radio,local_model_dropdown],
                                             [local_train_dataset_dropdown,local_val_dataset_dropdown,prefix1_textbox,
                                              datatset_col1_dropdown,prefix2_textbox,datatset_col2_dropdown,prefix3_textbox,
                                              datatset_col3_dropdown,prefix4_textbox,datatset_col4_dropdown])

    # train_file_upload_button.upload(fn=upload_train_file, inputs=[train_file_upload_button], outputs=[uploaded_train_file_path_textbox, uploaded_train_file_path_textbox,datatset_col1_dropdown,datatset_col2_dropdown])
    # val_file_upload_button.upload(fn=upload_val_file, inputs=[val_file_upload_button], outputs=[uploaded_val_file_path_textbox, uploaded_val_file_path_textbox])
    dataset_source_radio.change(fn=change_dataset_source, inputs=[dataset_source_radio,base_model_source_radio,base_model_name_dropdown,local_model_dropdown,hg_dataset_path_textbox],
                                outputs=[local_train_path_dataset_dropdown,local_train_dataset_dropdown,local_val_dataset_dropdown,
                                         hg_dataset_path_textbox,download_local_dataset_btn,local_train_path_dataset_dropdown,
                                         refresh_local_train_path_dataset_list_btn,local_train_dataset_dropdown,
                                         local_val_dataset_dropdown,download_dataset_status_markdown,stop_download_local_dataset_btn,
                                         hg_train_dataset_dropdown,hg_val_dataset_dropdown,prefix1_textbox,datatset_col1_dropdown,prefix2_textbox, datatset_col2_dropdown,prefix3_textbox,datatset_col3_dropdown,prefix4_textbox, datatset_col4_dropdown
                                         ])
    datatset_col1_dropdown.change(change_dataset_col1,[prefix1_textbox,datatset_col1_dropdown,prefix2_textbox,datatset_col2_dropdown
        ,prefix3_textbox,datatset_col3_dropdown,prefix4_textbox,datatset_col4_dropdown],[prompt_sample_textbox])
    datatset_col2_dropdown.change(change_dataset_col2, [prefix1_textbox,datatset_col1_dropdown,prefix2_textbox,datatset_col2_dropdown
        ,prefix3_textbox,datatset_col3_dropdown,prefix4_textbox,datatset_col4_dropdown], [prompt_sample_textbox])
    datatset_col3_dropdown.change(change_dataset_col3, [prefix1_textbox,datatset_col1_dropdown,prefix2_textbox,datatset_col3_dropdown
        ,prefix3_textbox,datatset_col3_dropdown,prefix4_textbox,datatset_col4_dropdown], [prompt_sample_textbox])
    datatset_col4_dropdown.change(change_dataset_col4, [prefix1_textbox,datatset_col1_dropdown,prefix2_textbox,datatset_col4_dropdown
        ,prefix3_textbox,datatset_col3_dropdown,prefix4_textbox,datatset_col4_dropdown], [prompt_sample_textbox])
    prefix1_textbox.change(change_prefix1_textbox,
                                  [prefix1_textbox, datatset_col1_dropdown, prefix2_textbox, datatset_col2_dropdown
                                      ,prefix3_textbox,datatset_col3_dropdown,prefix4_textbox,datatset_col4_dropdown],
                                  [prompt_sample_textbox])
    prefix2_textbox.change(change_prefix2_textbox,
                                  [prefix1_textbox, datatset_col1_dropdown, prefix2_textbox, datatset_col2_dropdown
                                      ,prefix3_textbox,datatset_col3_dropdown,prefix4_textbox,datatset_col4_dropdown],
                                  [prompt_sample_textbox])
    prefix3_textbox.change(change_prefix3_textbox,
                                  [prefix1_textbox, datatset_col1_dropdown, prefix3_textbox, datatset_col2_dropdown
                                      ,prefix3_textbox,datatset_col3_dropdown,prefix4_textbox,datatset_col4_dropdown],
                                  [prompt_sample_textbox])
    prefix4_textbox.change(change_prefix4_textbox,
                                  [prefix1_textbox, datatset_col1_dropdown, prefix4_textbox, datatset_col2_dropdown
                                      ,prefix3_textbox,datatset_col3_dropdown,prefix4_textbox,datatset_col4_dropdown],
                                  [prompt_sample_textbox])


    def download_model_postprocess():
        return gr.update(visible=True), gr.update(visible=False)
    def click_download_local_model_btn():
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)

    def click_stop_download_local_model_btn():
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

    def postprocess_click_download_local_model_btn(download_model_status_markdown):
        if download_model_status_markdown=="Model's name is empty!" or download_model_status_markdown=="Model has already been downloaded.":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=True), gr.update(visible=False)
    click_download_local_model_btn_event = download_local_model_btn.click(check_local_model_or_dataset_is_empty2,[base_model_name_dropdown,Huggingface_hub_token]).success(click_download_local_model_btn, [],
                                                                              [download_local_model_btn,
                                                                               stop_download_local_model_btn,
                                                                               download_model_status_markdown]).then(
        download_model_wrapper, [base_model_name_dropdown, local_model_root_dir_textbox],download_model_status_markdown) \
        .then(download_model_postprocess,[],[download_local_model_btn,stop_download_local_model_btn])
        # .then(
        # postprocess_click_download_local_model_btn,download_model_status_markdown,[download_local_model_btn,stop_download_local_model_btn])
    stop_download_local_model_btn.click(click_stop_download_local_model_btn, [],
                                          [download_local_model_btn, stop_download_local_model_btn,download_model_status_markdown],
                                          cancels=[click_download_local_model_btn_event])

    base_model_source_radio.change(change_base_model_source, [base_model_source_radio,base_model_name_dropdown,local_model_dropdown,local_train_path_dataset_dropdown],
                                   [base_model_name_dropdown,local_model_dropdown,refresh_local_model_list_btn,download_model_status_markdown,download_model_status_markdown,download_model_status_markdown,download_local_model_btn,stop_download_local_model_btn,
                                    prefix1_textbox, prefix2_textbox, prefix3_textbox, prefix4_textbox
                                    ],
                                   cancels=[click_download_local_model_btn_event])
    base_model_name_dropdown.change(change_base_model_name_dropdown, base_model_name_dropdown,
                                                                     [prefix1_textbox,prefix2_textbox,prefix3_textbox,prefix4_textbox,download_model_status_markdown,download_local_model_btn,stop_download_local_model_btn],cancels=[click_download_local_model_btn_event])

    def change_local_model_dropdown(local_model_dropdown,local_train_path_dataset_dropdown):

        global DATASET_FIRST_ROW
        local_dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets',local_train_path_dataset_dropdown)
        DATASET_FIRST_ROW, split_list = get_first_row_from_dataset(local_dataset_path)
        if "train" in split_list:
            split_list.pop(split_list.index("train"))
            split_list.insert(0, "train")

        col_names = list(DATASET_FIRST_ROW)

        new_col_names = []
        for col_name in col_names:
            if col_name.lower() != "id":
                new_col_names.append(col_name)
        INIT_COL1_TEXT = ""
        if len(new_col_names) > 0:
            INIT_COL1_TEXT = new_col_names[0]
        INIT_COL2_TEXT = ""
        if len(new_col_names) > 1:
            INIT_COL2_TEXT = new_col_names[1]
        INIT_COL3_TEXT = ""
        if len(new_col_names) > 2:
            INIT_COL3_TEXT = new_col_names[2]
        INIT_COL4_TEXT = ""
        if len(new_col_names) > 3:
            INIT_COL4_TEXT = new_col_names[3]
        col_names.insert(0, "")

        model_type = get_model_type(local_model_dropdown)

        if "instruction" in col_names and "input" in col_names and "output" in col_names:
            if model_type == "mistral" or model_type == "llama2":
                INIT_PREFIX1 = "<s>[INST] "
                INIT_PREFIX2 = "here are the inputs "
                INIT_PREFIX3 = " [/INST]"
                INIT_PREFIX4 = "</s>"
            elif model_type == "zephyr":
                INIT_PREFIX1 = "<|user|>\n"
                INIT_PREFIX2 = "here are the inputs "
                INIT_PREFIX3 = "</s><|assistant|>\n"
                INIT_PREFIX4 = "</s>"
            else:
                INIT_PREFIX1 = ""
                INIT_PREFIX2 = ""
                INIT_PREFIX3 = ""
                INIT_PREFIX4 = ""
        else:
            if model_type == "mistral" or model_type == "llama2":
                INIT_PREFIX1 = "<s>[INST] "
                INIT_PREFIX2 = " [/INST]"
                INIT_PREFIX3 = "</s>"
                INIT_PREFIX4 = ""
            elif model_type == "zephyr":
                INIT_PREFIX1 = "<|user|>\n"
                INIT_PREFIX2 = "</s><|assistant|>\n"
                INIT_PREFIX3 = "</s>"
                INIT_PREFIX4 = ""
            else:
                INIT_PREFIX1 = ""
                INIT_PREFIX2 = ""
                INIT_PREFIX3 = ""
                INIT_PREFIX4 = ""
        return INIT_PREFIX1, INIT_PREFIX2, INIT_PREFIX3, INIT_PREFIX4

    local_model_dropdown.change(change_local_model_dropdown, [local_model_dropdown,local_train_path_dataset_dropdown],
                                                                     [prefix1_textbox,prefix2_textbox,prefix3_textbox,prefix4_textbox])



    def check_training_params(prefix3_textbox,datatset_col3_dropdown,prefix4_textbox,datatset_col4_dropdown,
                              dataset_source_radio,hg_dataset_path_textbox,local_train_path_dataset_dropdown,local_train_dataset_dropdown,local_val_dataset_dropdown,
                        prefix1_textbox,datatset_col1_dropdown,prefix2_textbox,datatset_col2_dropdown,max_length_dropdown,
                        base_model_source_radio,local_model_dropdown,base_model_name_dropdown,fine_tuning_type_dropdown,
                        lora_r_slider,lora_alpha_slider,lora_dropout_slider,lora_bias_dropdown,epochs_slider,batch_size_slider,
                        learning_rate_slider,optimizer_dropdown,lr_scheduler_type_dropdown,gradient_checkpointing_checkbox,gradient_accumulation_steps_slider,
                        warmup_steps_slider,early_stopping_patience_slider,eval_steps_slider,hg_train_dataset_dropdown,hg_val_dataset_dropdown):
        if isinstance(hg_dataset_path_textbox, str):
            hg_dataset_path_textbox = hg_dataset_path_textbox.strip()
        if isinstance(local_model_dropdown,str):
            local_model_dropdown = local_model_dropdown.strip()
        if isinstance(max_length_dropdown,str):
            max_length_dropdown = max_length_dropdown.strip()
        if isinstance(base_model_name_dropdown, str):
            base_model_name_dropdown = base_model_name_dropdown.strip()
        if dataset_source_radio == "Download From Huggingface Hub":
            if not hg_dataset_path_textbox:
                raise gr.Error("Dataset's name is empty!")

            dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                       "datasets", hg_dataset_path_textbox)
            if not os.path.exists(dataset_dir):
                raise gr.Error(f"Dataset {dataset_dir} does not exist!")
            if not os.path.exists(os.path.join(dataset_dir, "dataset_infos.json")) and not os.path.exists(os.path.join(dataset_dir, "dataset_dict.json")):
                raise gr.Error(f"Invalid Dataset:{dataset_dir}!")
        else:
            if local_train_path_dataset_dropdown == "":
                raise gr.Error("Dataset's name is empty!")
            if not local_train_dataset_dropdown:
                raise gr.Error("Train dataset's name is empty!")
        if  not (datatset_col1_dropdown or datatset_col2_dropdown or datatset_col3_dropdown or datatset_col4_dropdown):
            raise gr.Error("At lease one column is required:{ColumnName1,ColumnName2,ColumnName3,ColumnName4}!")

        if max_length_dropdown !="Model Max Length":
            try:
                tempval = int(max_length_dropdown)
            except ValueError as e:
                raise gr.Error("Max Length is invalid!")
        if base_model_source_radio == "Download From Huggingface Hub":
            if not base_model_name_dropdown:
                raise gr.Error("Model Name is empty!")
            else:
                if not validate_model_path(base_model_name_dropdown)[0]:
                    raise gr.Error(f"Can't find model file:{base_model_name_dropdown}")
        else:
            if not local_model_dropdown:
                raise gr.Error("Model Name is empty!")

    def click_train_btn(prefix3_textbox,datatset_col3_dropdown,prefix4_textbox,datatset_col4_dropdown,
                        dataset_source_radio,hg_dataset_path_textbox,local_train_path_dataset_dropdown,local_train_dataset_dropdown,local_val_dataset_dropdown,
                        prefix1_textbox,datatset_col1_dropdown,prefix2_textbox,datatset_col2_dropdown,max_length_dropdown,
                        base_model_source_radio,local_model_dropdown,base_model_name_dropdown,fine_tuning_type_dropdown,
                        lora_r_slider,lora_alpha_slider,lora_dropout_slider,lora_bias_dropdown,epochs_slider,batch_size_slider,
                        learning_rate_slider,optimizer_dropdown,lr_scheduler_type_dropdown,gradient_checkpointing_checkbox,gradient_accumulation_steps_slider,
                        warmup_steps_slider,early_stopping_patience_slider,eval_steps_slider,hg_train_dataset_dropdown,hg_val_dataset_dropdown,training_log_progress = gr.Progress()):
        global train_param_config,infer_model
        global training_ret_val,error_msg

        global RAG_DATA_LIST_DROPDOWN, TEXT_SPLITTER_DROPDOWN, CHUNK_SIZE_SLIDER, CHUNK_OVERLAP_SLIDER, SEPARATORS_TEXTBOX
        global EMBEDDING_MODEL_SOURCE_RADIO, HUB_EMBEDDING_MODEL_NAMES_DROPDOWN, LOCAL_EMBEDDING_MODEL_NAMES_DROPDOWN
        global CHAT_MODEL_SOURCE_RADIO, HUB_CHAT_MODEL_NAMES_DROPDOWN, LOCAL_CHAT_MODEL_NAMES_DROPDOWN
        global qa_with_rag, SEARCH_TOP_K_SLIDER, SEARCH_SCORE_THRESHOLD_SLIDER

        if infer_model:
            infer_model.free_memory()
            infer_model = None
        qa_with_rag.free_memory()
        torch.cuda.empty_cache()

        RAG_DATA_LIST_DROPDOWN = ""
        TEXT_SPLITTER_DROPDOWN = ""
        CHUNK_SIZE_SLIDER = 0
        CHUNK_OVERLAP_SLIDER = -1
        SEPARATORS_TEXTBOX = ""
        EMBEDDING_MODEL_SOURCE_RADIO = ""
        HUB_EMBEDDING_MODEL_NAMES_DROPDOWN = ""
        LOCAL_EMBEDDING_MODEL_NAMES_DROPDOWN = ""
        CHAT_MODEL_SOURCE_RADIO = ""
        HUB_CHAT_MODEL_NAMES_DROPDOWN = ""
        LOCAL_CHAT_MODEL_NAMES_DROPDOWN = ""
        SEARCH_TOP_K_SLIDER = ""
        SEARCH_SCORE_THRESHOLD_SLIDER = ""

        if isinstance(hg_dataset_path_textbox, str):
            hg_dataset_path_textbox = hg_dataset_path_textbox.strip()
        if isinstance(local_model_dropdown,str):
            local_model_dropdown = local_model_dropdown.strip()
        if isinstance(max_length_dropdown,str):
            max_length_dropdown = max_length_dropdown.strip()
        if isinstance(base_model_name_dropdown, str):
            base_model_name_dropdown = base_model_name_dropdown.strip()
        if isinstance(local_val_dataset_dropdown, str):
            local_val_dataset_dropdown = local_val_dataset_dropdown.strip()

        # dataset config
        if dataset_source_radio == "Download From Huggingface Hub":
            train_param_config["dataset"]["is_huggingface_dataset"] = True
        else:
            train_param_config["dataset"]["is_huggingface_dataset"] = False

        train_param_config["dataset"]["hg_train_dataset"] = hg_train_dataset_dropdown
        train_param_config["dataset"]["hg_val_dataset"] = hg_val_dataset_dropdown

        train_param_config["dataset"]["huggingface_dataset_name"] = hg_dataset_path_textbox
        train_param_config["dataset"]["local_dataset_name"] = local_train_path_dataset_dropdown
        train_param_config["dataset"]["local_train_set"] = local_train_dataset_dropdown
        train_param_config["dataset"]["local_val_set"] = local_val_dataset_dropdown
        train_param_config["dataset"]["prefix1"] = prefix1_textbox
        train_param_config["dataset"]["prefix2"] = prefix2_textbox
        train_param_config["dataset"]["prefix3"] = prefix3_textbox
        train_param_config["dataset"]["prefix4"] = prefix4_textbox
        train_param_config["dataset"]["datatset_col1"] = datatset_col1_dropdown
        train_param_config["dataset"]["datatset_col2"] = datatset_col2_dropdown
        train_param_config["dataset"]["datatset_col3"] = datatset_col3_dropdown
        train_param_config["dataset"]["datatset_col4"] = datatset_col4_dropdown
        if max_length_dropdown !="Model Max Length":
            try:
                train_param_config["dataset"]["max_length"] = int(max_length_dropdown)
            except ValueError as e:
                raise gr.Error("Max Length is invalid!")
        else:
            train_param_config["dataset"]["max_length"] = max_length_dropdown
        # model config
        if base_model_source_radio == "Download From Huggingface Hub":
            if base_model_name_dropdown == "":
                raise gr.Error("Model Name is empty!")
        else:
            if local_model_dropdown == "":
                raise gr.Error("Model Name is empty!")
        train_param_config["model"]["base_model_source"] = base_model_source_radio
        train_param_config["model"]["local_model"] = local_model_dropdown
        train_param_config["model"]["base_model_name"] = base_model_name_dropdown
        train_param_config["model"]["fine_tuning_type"] = fine_tuning_type_dropdown
        train_param_config["model"]["lora_dropout"] = lora_dropout_slider
        train_param_config["model"]["lora_bias"] = lora_bias_dropdown
        train_param_config["model"]["lora_r"] = lora_r_slider
        train_param_config["model"]["lora_alpha"] = lora_alpha_slider

        # training config
        train_param_config["training"]["epochs"] = epochs_slider
        train_param_config["training"]["batch_size"] = batch_size_slider
        train_param_config["training"]["learning_rate"] = learning_rate_slider
        train_param_config["training"]["optimizer"] = optimizer_dropdown
        train_param_config["training"]["gradient_checkpointing"] = gradient_checkpointing_checkbox
        train_param_config["training"]["gradient_accumulation_steps"] = gradient_accumulation_steps_slider
        train_param_config["training"]["warmup_steps"] = warmup_steps_slider
        train_param_config["training"]["early_stopping_patience"] = early_stopping_patience_slider
        train_param_config["training"]["eval_steps"] = eval_steps_slider
        train_param_config["training"]["lr_scheduler_type"] = lr_scheduler_type_dropdown


        if train_param_config["dataset"]["is_huggingface_dataset"]:
            train_param_config["dataset"]["hg_dataset_dir"] = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                       "datasets", train_param_config["dataset"]["huggingface_dataset_name"])
        else:
            train_param_config["dataset"]["hg_dataset_dir"] = None
            train_param_config["dataset"]["local_dataset_dir"] = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                       "datasets", train_param_config["dataset"]["local_dataset_name"])
        if train_param_config["model"]["base_model_source"] == "Download From Huggingface Hub":
            base_model_name = train_param_config["model"]["base_model_name"]
        else:
            base_model_name = train_param_config["model"]["local_model"]
        train_param_config["model"]["base_model_name"] = base_model_name
        train_param_config["model"]["base_model_path"] = os.path.join(os.path.dirname(os.path.abspath(__file__)),"models",base_model_name)
        train_param_config["training"]["root_dir"] = os.path.dirname(os.path.abspath(__file__))

        TRAINING_STATUS.status = 0
        qlora_trainer = QloraTrainer(train_param_config)
        training_log_progress(0.0)
        yield "### &nbsp;&nbsp;Loading model ...", gr.update(), gr.update()
        load_model_status,msg = qlora_trainer.load_model()
        if load_model_status == -1:
            raise gr.Error(f"Loading model error:{msg}")
            if infer_model:
                infer_model.free_memory()
                infer_model = None
            torch.cuda.empty_cache()
            return
        training_log_progress(0.0)
        yield "### &nbsp;&nbsp;Loading dataset ...", gr.update(), gr.update()
        qlora_trainer.load_dataset()
        training_log_progress(0.0)
        yield "### &nbsp;&nbsp;Start training ...", gr.update(), gr.update()
        #
        def training_thread_fun():
            global training_ret_val,error_msg
            training_ret_val,error_msg = qlora_trainer.train()

        training_thread = threading.Thread(target=training_thread_fun)
        training_thread.start()
        last_step = -1
        start_time = -1
        while training_thread.is_alive():
            time.sleep(1.0)

            remaining_time = 0
            elapsed_time = 0
            try:
                if qlora_trainer.logging_callback.max_steps>0:
                    if last_step >= 0:
                        elapsed_steps = qlora_trainer.logging_callback.current_step - last_step
                        if elapsed_steps >= 5 and start_time >= 0:
                            elapsed_time = time.time() - start_time
                            time_per_step = elapsed_time/elapsed_steps
                            remaining_time = (qlora_trainer.logging_callback.max_steps - qlora_trainer.logging_callback.current_step)*time_per_step

                    else:
                        last_step = qlora_trainer.logging_callback.current_step
                        start_time = time.time()
                    progress_val = qlora_trainer.logging_callback.current_step / qlora_trainer.logging_callback.max_steps
                    training_log_progress(progress_val)
                    elapsed_time_int = int(elapsed_time)
                    remaining_time_int = int(remaining_time)
                    yield   f"### &nbsp;&nbsp;Training: {progress_val*100:0.0f}% | {qlora_trainer.logging_callback.current_step} /{qlora_trainer.logging_callback.max_steps}[{elapsed_time_int//60}:{elapsed_time_int%60}<{remaining_time_int//60}:{remaining_time_int%60}]", gr.update(), gr.update()
            except:
                pass

        # print(train_param_config)

        if training_ret_val == 0:
            training_log_progress(1.0, desc="Mergeing base model with adapter...")
            yield f"### &nbsp;&nbsp;Mergeing base model with adapter...", gr.update(), gr.update()
            qlora_trainer.merge_and_save()
            training_log_progress(1.0)
            TRAINING_STATUS.status = 2
            qlora_trainer.free_memroy()
            runs_model_names = get_runs_models()
            yield f"### &nbsp;&nbsp;<span style='color:green'>Training process is over: {qlora_trainer.logging_callback.current_step} /{qlora_trainer.logging_callback.max_steps}</span>", gr.update(choices=runs_model_names,value=runs_model_names[0] if runs_model_names else None), gr.update(choices=runs_model_names,value=runs_model_names[0] if runs_model_names else None)
        else:
            TRAINING_STATUS.status = 2
            qlora_trainer.free_memroy()
            runs_model_names = get_runs_models()
            yield f"### &nbsp;&nbsp;<span style='color:red'>Training is interrupted: {error_msg}</span>", gr.update(choices=runs_model_names,value=runs_model_names[0] if runs_model_names else None), gr.update(choices=runs_model_names,value=runs_model_names[0] if runs_model_names else None)


        # try:
        #     device = cuda.get_current_device()
        #     device.reset()
        # except Exception as e:
        #     pass

        #
    def update_runs_output_model_dropdown():
        training_runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')
        run_names = os.listdir(training_runs_dir)
        run_names.sort(key=lambda file: os.path.getmtime(os.path.join(training_runs_dir, file)))
        runs_output_model = []
        for run_name in run_names:
            run_name_dir = os.path.join(training_runs_dir, run_name)
            run_output_model = os.path.join(run_name_dir, "output_model")
            if os.path.exists(run_output_model):
                run_output_model_names = os.listdir(run_output_model)
                for run_output_model_name in run_output_model_names:
                    model_bin_path = os.path.exists(
                        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs',
                                     run_name, "output_model", run_output_model_name, "ori",
                                     "pytorch_model.bin"))
                    if run_output_model_name.find("merged_") >= 0 and model_bin_path:
                        runs_output_model.append(os.path.join(run_name, "output_model", run_output_model_name, "ori"))
        runs_output_model = runs_output_model[::-1]
        return gr.update(choices=runs_output_model,value=runs_output_model[0] if runs_output_model else "")

    train_params = [prefix3_textbox,datatset_col3_dropdown,prefix4_textbox,datatset_col4_dropdown,dataset_source_radio,
     hg_dataset_path_textbox,
     local_train_path_dataset_dropdown,
     local_train_dataset_dropdown,
     local_val_dataset_dropdown,
     prefix1_textbox,
     datatset_col1_dropdown,
     prefix2_textbox,
     datatset_col2_dropdown,
     max_length_dropdown,
     base_model_source_radio,
     local_model_dropdown,
     base_model_name_dropdown,
     fine_tuning_type_dropdown,
     lora_r_slider,
     lora_alpha_slider,
     lora_dropout_slider,
     lora_bias_dropdown,
     epochs_slider,
     batch_size_slider,
     learning_rate_slider,
     optimizer_dropdown,
     lr_scheduler_type_dropdown,
     gradient_checkpointing_checkbox,
     gradient_accumulation_steps_slider,
     warmup_steps_slider,
     early_stopping_patience_slider,
     eval_steps_slider,
     hg_train_dataset_dropdown,
     hg_val_dataset_dropdown,
     ]

    def change_to_tensorboard_tab():
        # ping_tensorboard_cmd = "nc -zv localhost 6006"
        try:
            # ping_output = subprocess.check_output(ping_tensorboard_cmd, shell=True)
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((LOCAL_HOST_IP, 6006))
            s.close()
        except Exception as e:
            return gr.update(
                value="The tensorboard could be temporarily unavailable. Try switch back in a few moments.")
        sss = np.random.randint(0, 100) + 1000
        iframe = f'<iframe src={TENSORBOARD_URL} style="border:none;height:{sss}px;width:100%">'
        return gr.Tabs.update(selected=1),gr.update(value=iframe)
    def set_test_prompt(prefix1_textbox,prefix2_textbox):
        return gr.update(value=prefix1_textbox),gr.update(value=prefix2_textbox)

    def update_test_input_prompt(prefix1_textbox,datatset_col1_dropdown,prefix2_textbox,datatset_col2_dropdown,
                                 prefix3_textbox,datatset_col3_dropdown,prefix4_textbox,datatset_col4_dropdown
                                 ):
        output = ""
        if prefix4_textbox:
            output = prefix1_textbox+prefix2_textbox+prefix3_textbox
        elif prefix3_textbox:
            output = prefix1_textbox+prefix2_textbox
        elif prefix2_textbox:
            output = prefix1_textbox
        return output

    train_btn.click(update_test_input_prompt,[prefix1_textbox,datatset_col1_dropdown,prefix2_textbox,datatset_col2_dropdown,
                    prefix3_textbox,datatset_col3_dropdown,prefix4_textbox,datatset_col4_dropdown],test_input_textbox).\
        then(check_training_params,train_params).success(change_to_tensorboard_tab,None,[tensorboard_tab,tensorboard_html]).then(click_train_btn,train_params,[training_log_markdown,runs_output_model_dropdown,quantization_runs_output_model_dropdown])
    refresh_local_model_list_btn.click(click_refresh_local_model_list_btn, [refresh_local_model_list_btn], local_model_dropdown)
    delete_text_btn.click(click_delete_text_btn, training_runs_dropdown, [training_runs_dropdown, tensorboard_html])

    ###############################
    def update_prompt_template(hg_dataset_path_textbox, base_model_name_dropdown, base_model_source_radio,
                               local_model_dropdown):
        global DATASET_FIRST_ROW
        hg_dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', hg_dataset_path_textbox)
        DATASET_FIRST_ROW, split_list = get_first_row_from_dataset(hg_dataset_path)
        col_names = list(DATASET_FIRST_ROW)

        new_col_names = []
        for col_name in col_names:
            if col_name.lower() != "id":
                new_col_names.append(col_name)
        INIT_COL1_TEXT = ""
        if len(new_col_names) > 0:
            INIT_COL1_TEXT = new_col_names[0]
        INIT_COL2_TEXT = ""
        if len(new_col_names) > 1:
            INIT_COL2_TEXT = new_col_names[1]
        INIT_COL3_TEXT = ""
        if len(new_col_names) > 2:
            INIT_COL3_TEXT = new_col_names[2]
        INIT_COL4_TEXT = ""
        if len(new_col_names) > 3:
            INIT_COL4_TEXT = new_col_names[3]
        col_names.insert(0, "")
        if base_model_source_radio == "Download From Huggingface Hub":
            curr_model_name = base_model_name_dropdown
        else:
            curr_model_name = local_model_dropdown
        model_type = get_model_type(curr_model_name)
        if not curr_model_name or model_type == "other model":
            return "", gr.update(choices=col_names, value=INIT_COL1_TEXT), "", gr.update(choices=col_names,
                                                                                         value=INIT_COL2_TEXT), "", gr.update(
                choices=col_names, value=INIT_COL3_TEXT), "", gr.update(choices=col_names, value=INIT_COL4_TEXT),
        else:
            if "instruction" in col_names and "input" in col_names and "output" in col_names:
                if model_type == "mistral" or model_type == "llama2":
                    INIT_PREFIX1 = "<s>[INST] "
                    INIT_PREFIX2 = "here are the inputs "
                    INIT_PREFIX3 = " [/INST]"
                    INIT_PREFIX4 = "</s>"
                else:
                    INIT_PREFIX1 = "<|user|>\n"
                    INIT_PREFIX2 = "here are the inputs "
                    INIT_PREFIX3 = "</s><|assistant|>\n"
                    INIT_PREFIX4 = "</s>"
                return INIT_PREFIX1, gr.update(choices=col_names, value=col_names[
                    col_names.index("instruction")]), INIT_PREFIX2, gr.update(choices=col_names, value=col_names[
                    col_names.index("input")]), INIT_PREFIX3, gr.update(choices=col_names, value=col_names[
                    col_names.index("output")]), INIT_PREFIX4, gr.update(choices=col_names, value="")
            else:
                if model_type == "mistral" or model_type == "llama2":
                    INIT_PREFIX1 = "<s>[INST] "
                    INIT_PREFIX2 = " [/INST]"
                    INIT_PREFIX3 = "</s>"
                    INIT_PREFIX4 = ""
                else:
                    INIT_PREFIX1 = "<|user|>\n"
                    INIT_PREFIX2 = "</s><|assistant|>\n"
                    INIT_PREFIX3 = "</s>"
                    INIT_PREFIX4 = ""
                return INIT_PREFIX1, gr.update(choices=col_names, value=INIT_COL1_TEXT), INIT_PREFIX2, gr.update(
                    choices=col_names, value=INIT_COL2_TEXT), INIT_PREFIX3, gr.update(choices=col_names,
                                                                                      value=INIT_COL3_TEXT), INIT_PREFIX4, gr.update(
                    choices=col_names, value=INIT_COL4_TEXT)


    click_download_local_dataset_btn_event = download_local_dataset_btn.click(check_local_model_or_dataset_is_empty3,
                                                                              [hg_dataset_path_textbox,Huggingface_hub_token]).success(
        click_download_local_dataset_btn, [],
        [download_local_dataset_btn, stop_download_local_dataset_btn, download_dataset_status_markdown]). \
        then(download_dataset_wrapper, [hg_dataset_path_textbox, local_dataset_root_dir_textbox],
             download_dataset_status_markdown, show_progress=True). \
        success(get_hg_dataset_train_val_set, [hg_dataset_path_textbox, local_dataset_root_dir_textbox],
                [hg_train_dataset_dropdown, hg_val_dataset_dropdown, download_local_dataset_btn,
                 stop_download_local_dataset_btn]). \
        success(update_prompt_template,
                [hg_dataset_path_textbox, base_model_name_dropdown, base_model_source_radio, local_model_dropdown],
                [prefix1_textbox, datatset_col1_dropdown, prefix2_textbox, datatset_col2_dropdown, prefix3_textbox,
                 datatset_col3_dropdown, prefix4_textbox, datatset_col4_dropdown])
    stop_download_local_dataset_btn.click(click_stop_download_local_dataset_btn, [],
                                          [download_local_dataset_btn, stop_download_local_dataset_btn,
                                           download_dataset_status_markdown],
                                          cancels=[click_download_local_dataset_btn_event])
    refresh_local_train_path_dataset_list_btn.click(click_refresh_local_train_dataset_list_btn, [],
                                                    local_train_path_dataset_dropdown)


    def click_generate_text_btn(runs_output_model_dropdown, test_input_textbox, max_new_tokens_slider,
                                       temperature_slider, top_k_slider, top_p_slider, repeat_penalty_slider,
                                       chat_history_window_slider,finetune_test_using_4bit_quantization_checkbox,
                                low_cpu_mem_usage_checkbox
                                ):
        global infer_model
        if infer_model:
            infer_model.free_memory()
            infer_model = None
        torch.cuda.empty_cache()
        if runs_output_model_dropdown:
            if test_input_textbox.strip():
                output_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs',
                                                 runs_output_model_dropdown)
                load_model_status = 0
                if output_model_path.split('.')[-1] == "gguf":
                    infer_model = LlamaCppInference(model_path=output_model_path, max_new_tokens=max_new_tokens_slider,
                                            temperature=temperature_slider,
                                            top_k=top_k_slider, top_p=top_p_slider,
                                            repetition_penalty=repeat_penalty_slider)
                    load_model_status, msg = infer_model.load_model()
                else:
                    infer_model = HuggingfaceInference(model_path=output_model_path, max_new_tokens=max_new_tokens_slider,
                                            temperature=temperature_slider,
                                            top_k=top_k_slider, top_p=top_p_slider,
                                            repetition_penalty=repeat_penalty_slider,
                                                       using_4bit_quantization=finetune_test_using_4bit_quantization_checkbox,
                                                       low_cpu_mem_usage=low_cpu_mem_usage_checkbox)
                    load_model_status, msg = infer_model.load_model()

                model_type = get_model_type(output_model_path)
                prompt_template = get_model_prompt_template(model_type)

                if load_model_status == -1:
                    raise gr.Error(f"Loading model error:{msg}")
                    if infer_model:
                        infer_model.free_memory()
                        infer_model = None
                    torch.cuda.empty_cache()
                    return
                input_prompt = prompt_template.format(question=test_input_textbox)
                # input_prompt = prompt_template(test_input_textbox)
                # print("s1:", model_type, input_prompt)
                gen_output_text = infer_model.infer(input_prompt)
                return gr.update(value=gen_output_text)
            else:
                raise gr.Error("Input is empty!")
        else:
            raise gr.Error("Model is empty!")


    generate_text_btn.click(click_generate_text_btn, [runs_output_model_dropdown, test_input_textbox, max_new_tokens_slider,
                                       temperature_slider, top_k_slider, top_p_slider, repeat_penalty_slider,
                                       chat_history_window_slider,finetune_test_using_4bit_quantization_checkbox,low_cpu_mem_usage_checkbox], test_output)


    def click_quantize_btn(quantization_type_dropdown, quantization_runs_output_model_dropdown,
                           prefix1_textbox, prefix2_textbox, datatset_col1_dropdown, datatset_col2_dropdown,
                           local_quantization_dataset_dropdown,
                           quantization_progress=gr.Progress()):
        quantization_progress(0.35)
        yield "Prepare model and dataset..."
        model_name = quantization_runs_output_model_dropdown.split(os.sep)[-2].split('_')[-1]

        quantized_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs',
                                           os.sep.join(quantization_runs_output_model_dropdown.split(os.sep)[0:-1]),
                                           "quantized_" + quantization_type_dropdown + "_" + model_name
                                           )
        if not os.path.exists(quantized_model_dir):
            os.makedirs(quantized_model_dir)
        if quantization_type_dropdown == "gptq":
            dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "datasets", local_quantization_dataset_dropdown)
            if not os.path.exists(dataset_path):
                raise gr.Error(f"Dataset does not exist:{dataset_path}")
                yield f"Dataset does not exist:{dataset_path}"
            run_output_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "runs", quantization_runs_output_model_dropdown)
            yield "Quantizing model..."
            torch.cuda.empty_cache()
            gptq = GptqLlmQuantizer(run_output_model_path, dataset_path, prefix1_textbox, prefix2_textbox,
                                    datatset_col1_dropdown, datatset_col2_dropdown)
            load_model_status, msg = gptq.load_model()
            if load_model_status == -1:
                raise gr.Error(msg)
                return ""
            gptq_dataset = gptq.prepare_dataset()
            gptq.quantize(gptq_dataset, quantized_model_dir)
            del gptq
            quantization_progress(1.0)
            yield f'<span style="color:green">&nbsp;&nbsp;&nbsp;&nbsp;Model is quantized to:{quantized_model_dir}</span>'

        else:
            pass




    # def change_runs_output_model_dropdown():
    #     runs_model_names = get_runs_models()
    #     return gr.update(choices=runs_model_names, value=runs_model_names[0] if runs_model_names else None)
    #
    # runs_output_model_dropdown.change(change_runs_output_model_dropdown,[],runs_output_model_dropdown)


    def change_quantization_runs_output_model_dropdown(quantization_runs_output_model_dropdown, quantization_type_dropdown):
        # model_name = quantization_runs_output_model_dropdown.split(os.sep)[-2].split('_')[-1]
        # quantized_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs',
        #                                    os.sep.join(quantization_runs_output_model_dropdown.split(os.sep)[0:-1]),
        #                                    "quantized_" + quantization_type_dropdown + "_" + model_name)
        # if not os.path.exists(quantized_model_dir):
        #     os.makedirs(quantized_model_dir)
        return gr.update(value=f"&nbsp;&nbsp;&nbsp;&nbsp;2.Convert {quantization_runs_output_model_dropdown} to gguf model")


    quantization_runs_output_model_dropdown.change(change_quantization_runs_output_model_dropdown,
                                                   [quantization_runs_output_model_dropdown, quantization_type_dropdown],
                                                   gguf_quantization_markdown2)


    def change_quantization_type_dropdown(quantization_type_dropdown, quantization_runs_output_model_dropdown):
        if quantization_type_dropdown == "gguf":
            model_name = quantization_runs_output_model_dropdown.split(os.sep)[-2].split('_')[-1]
            quantized_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               'runs',
                                               os.sep.join(
                                                   quantization_runs_output_model_dropdown.split(
                                                       os.sep)[0:-1]),
                                               "quantized_" + quantization_type_dropdown + "_" + model_name
                                               )
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), \
                   gr.update(visible=False), gr.update(visible=False), \
                   gr.update(value=f"&nbsp;&nbsp;&nbsp;&nbsp;2.Copy your gguf model to {quantized_model_dir}."), gr.update(
                visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
        else:
            return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(
                visible=True), gr.update(visible=True), gr.update(), gr.update(visible=False), gr.update(
                visible=False), gr.update(visible=False), gr.update(visible=False)


    quantization_type_dropdown.change(change_quantization_type_dropdown,
                                      [quantization_type_dropdown, quantization_runs_output_model_dropdown],
                                      [local_quantization_dataset_dropdown,
                                       refresh_local_quantization_dataset_btn,
                                       quantization_runs_output_model_dropdown, quantize_btn,
                                       quantization_logging_markdown, gguf_quantization_markdown2,
                                       gguf_quantization_markdown1, gguf_quantization_markdown2,
                                       gguf_quantization_markdown3, gguf_quantization_markdown0]
                                      )


    def click_refresh_deployment_runs_output_model_btn1(deployment_framework_dropdown):
        training_runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')
        run_names = os.listdir(training_runs_dir)
        run_names.sort(
            key=lambda file: os.path.getmtime(os.path.join(training_runs_dir, file)))
        # ori_model_runs_output_model = []
        tgi_model_format_runs_output_model = []
        gguf_model_format_runs_output_model = []
        for run_name in run_names:
            run_name_dir = os.path.join(training_runs_dir, run_name)
            run_output_model = os.path.join(run_name_dir, "output_model")
            if os.path.exists(run_output_model):
                run_output_model_names = os.listdir(run_output_model)
                for run_output_model_name in run_output_model_names:
                    model_bin_path = os.path.exists(
                        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs',
                                     run_name, "output_model", run_output_model_name, "ori",
                                     "pytorch_model.bin"))
                    if run_output_model_name.find("merged_") >= 0 and model_bin_path:
                        tgi_model_format_runs_output_model.append(
                            os.path.join(run_name, "output_model", run_output_model_name,
                                         "ori"))

                        gptq_model_path = os.path.join(
                            os.path.dirname(os.path.abspath(__file__)), 'runs', run_name,
                            "output_model", run_output_model_name,
                            "quantized_gptq_" + run_output_model_name.split('_')[-1],
                            "pytorch_model.bin")
                        if os.path.exists(gptq_model_path):
                            tgi_model_format_runs_output_model.append(
                                os.path.join(run_name, "output_model",
                                             run_output_model_name, "quantized_gptq_" +
                                             run_output_model_name.split('_')[-1]))
                        gguf_model_dir = os.path.join(
                            os.path.dirname(os.path.abspath(__file__)), 'runs', run_name,
                            "output_model", run_output_model_name,
                            "quantized_gguf_" + run_output_model_name.split('_')[-1])
                        if os.path.exists(gguf_model_dir):
                            gguf_model_names = os.listdir(gguf_model_dir)
                            for gguf_model_name in gguf_model_names:
                                if gguf_model_name.split('.')[-1] == "gguf":
                                    gguf_model_format_runs_output_model.append(
                                        os.path.join(run_name, "output_model",
                                                     run_output_model_name,
                                                     "quantized_gguf_" +
                                                     run_output_model_name.split('_')[-1],
                                                     gguf_model_name))
        if deployment_framework_dropdown == "TGI":
            tgi_model_format_runs_output_model = tgi_model_format_runs_output_model[::-1]
            return gr.update(choices=tgi_model_format_runs_output_model,
                             value=tgi_model_format_runs_output_model[0] if tgi_model_format_runs_output_model else "")
        else:
            gguf_model_format_runs_output_model = gguf_model_format_runs_output_model[::-1]
            return gr.update(choices=gguf_model_format_runs_output_model,
                             value=gguf_model_format_runs_output_model[0] if gguf_model_format_runs_output_model else "")
    def click_refresh_deployment_runs_output_model_btn2(deployment_framework_dropdown):
        training_runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')
        run_names = os.listdir(training_runs_dir)
        run_names.sort(
            key=lambda file: os.path.getmtime(os.path.join(training_runs_dir, file)))
        # ori_model_runs_output_model = []
        tgi_model_format_runs_output_model = []
        gguf_model_format_runs_output_model = []
        for run_name in run_names:
            run_name_dir = os.path.join(training_runs_dir, run_name)
            run_output_model = os.path.join(run_name_dir, "output_model")
            if os.path.exists(run_output_model):
                run_output_model_names = os.listdir(run_output_model)
                for run_output_model_name in run_output_model_names:
                    model_bin_path = os.path.exists(
                        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs',
                                     run_name, "output_model", run_output_model_name, "ori",
                                     "pytorch_model.bin"))
                    if run_output_model_name.find("merged_") >= 0 and model_bin_path:
                        tgi_model_format_runs_output_model.append(
                            os.path.join(run_name, "output_model", run_output_model_name,
                                         "ori"))

                        gptq_model_path = os.path.join(
                            os.path.dirname(os.path.abspath(__file__)), 'runs', run_name,
                            "output_model", run_output_model_name,
                            "quantized_gptq_" + run_output_model_name.split('_')[-1],
                            "pytorch_model.bin")
                        if os.path.exists(gptq_model_path):
                            tgi_model_format_runs_output_model.append(
                                os.path.join(run_name, "output_model",
                                             run_output_model_name, "quantized_gptq_" +
                                             run_output_model_name.split('_')[-1]))
                        gguf_model_dir = os.path.join(
                            os.path.dirname(os.path.abspath(__file__)), 'runs', run_name,
                            "output_model", run_output_model_name,
                            "quantized_gguf_" + run_output_model_name.split('_')[-1])
                        if os.path.exists(gguf_model_dir):
                            gguf_model_names = os.listdir(gguf_model_dir)
                            for gguf_model_name in gguf_model_names:
                                if gguf_model_name.split('.')[-1] == "gguf":
                                    gguf_model_format_runs_output_model.append(
                                        os.path.join(run_name, "output_model",
                                                     run_output_model_name,
                                                     "quantized_gguf_" +
                                                     run_output_model_name.split('_')[-1],
                                                     gguf_model_name))
        if deployment_framework_dropdown == "TGI":
            tgi_model_format_runs_output_model = tgi_model_format_runs_output_model[::-1]
            return gr.update(choices=tgi_model_format_runs_output_model,
                             value=tgi_model_format_runs_output_model[0] if tgi_model_format_runs_output_model else "")
        else:
            gguf_model_format_runs_output_model = gguf_model_format_runs_output_model[::-1]
            return gr.update(choices=gguf_model_format_runs_output_model,
                             value=gguf_model_format_runs_output_model[0] if gguf_model_format_runs_output_model else "")


    refresh_deployment_runs_output_model_btn.click(click_refresh_deployment_runs_output_model_btn1,
                                                   [deployment_framework_dropdown], deployment_runs_output_model_dropdown)
    quantize_btn.click(click_quantize_btn,
                       [quantization_type_dropdown, quantization_runs_output_model_dropdown,
                        prefix1_textbox, prefix2_textbox, datatset_col1_dropdown,
                        datatset_col2_dropdown,
                        local_quantization_dataset_dropdown
                        ], quantization_logging_markdown).success(
        click_refresh_deployment_runs_output_model_btn2, [deployment_framework_dropdown],
        deployment_runs_output_model_dropdown)


    def change_deployment_runs_output_model_dropdown(deployment_framework_dropdown, deployment_runs_output_model_dropdown):
        if deployment_framework_dropdown == "TGI":
            if deployment_runs_output_model_dropdown:
                model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs',
                                         os.path.dirname(deployment_runs_output_model_dropdown))
                model_name = os.path.basename(deployment_runs_output_model_dropdown)

                if model_name.rfind("quantized_gptq_") >= 0:
                    run_server_value = f'''docker run --gpus all --shm-size 1g -p 8080:80 -v {model_dir}:/data ghcr.io/huggingface/text-generation-inference:latest --model-id /data/{model_name} --quantize gptq'''
                else:
                    run_server_value = f'''docker run --gpus all --shm-size 1g -p 8080:80 -v {model_dir}:/data ghcr.io/huggingface/text-generation-inference:latest --model-id /data/{model_name}'''
                # run_server_value = f'''sh deploy_llm.sh 8080 {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', deployment_runs_output_model_dropdown)}'''
                run_client_value = '''Command-Line Interface(CLI):\ncurl 127.0.0.1:8080/generate -X POST  -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":20}}' -H 'Content-Type: application/json'\n\nPython:\nfrom huggingface_hub import InferenceClient \nclient = InferenceClient(model="http://127.0.0.1:8080")\noutput = client.text_generation(prompt="What is Deep Learning?",max_new_tokens=512)
                                                '''
            else:
                run_server_value = ""
                run_client_value = ""
            return gr.update(), gr.update(value=run_server_value), gr.update(value=run_client_value)
        elif deployment_framework_dropdown == "llama-cpp-python":
            if deployment_runs_output_model_dropdown.rfind('_gguf_') >= 0:
                code_str = f'''
        from llama_cpp import Llama
        model_path = '{os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', deployment_runs_output_model_dropdown)}'
        llm = Llama(model_path)
        output = llm("Q: Name the planets in the solar system? A: ", stop=["\\n"], echo=True)
        print("output:",output['choices'][0]['text'])
                                            '''
                return gr.update(value=code_str), gr.update(), gr.update()
            else:
                # gr.Warning("llama-cpp-python only support gguf format!")

                return gr.update(value="llama-cpp-python only support gguf format!"), gr.update(), gr.update()


    deployment_runs_output_model_dropdown.change(change_deployment_runs_output_model_dropdown,
                                                 [deployment_framework_dropdown, deployment_runs_output_model_dropdown],
                                                 [run_llama_cpp_python_code, run_server_script_textbox,
                                                  run_client_script_textbox])


    def change_deployment_framework_dropdown(deployment_framework_dropdown, deployment_runs_output_model_dropdown):
        if deployment_framework_dropdown == "TGI":
            return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
        elif deployment_framework_dropdown == "llama-cpp-python":
            if deployment_runs_output_model_dropdown and deployment_runs_output_model_dropdown.rfind('_gguf_') >= 0:
                code_str = f'''
        from llama_cpp import Llama
        model_path = '{os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', deployment_runs_output_model_dropdown)}'
        output = llm("Q: Name the planets in the solar system? A: ", stop=["\\n"], echo=True)
        print("output:",output['choices'][0]['text'])
        '''
                return gr.update(value=code_str, visible=True), gr.update(visible=False), gr.update(
                    visible=False), gr.update(visible=False)
            elif deployment_runs_output_model_dropdown:
                # gr.Warning("llama-cpp-python only support gguf format!")
                return gr.update(value="llama-cpp-python only support gguf format!", visible=True), gr.update(
                    visible=False), gr.update(visible=False), gr.update(visible=False)
            else:
                return gr.update(value="runs_output_model is empty!", visible=True), gr.update(
                    visible=False), gr.update(visible=False), gr.update(visible=False)


    deployment_framework_dropdown.change(change_deployment_framework_dropdown,
                                         [deployment_framework_dropdown, deployment_runs_output_model_dropdown],
                                         [run_llama_cpp_python_code, run_server_script_textbox, run_client_script_textbox,
                                          install_requirements_accordion])

    ###################
    ##################

    def download_hub_embedding_model_postprocess():
        return gr.update(visible=True), gr.update(visible=False)


    def click_download_hub_embedding_model_btn():
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)


    def click_stop_download_hub_embedding_model_names_btn():
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)


    def click_stop_download_hub_embedding_model_names_btn():
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)


    def change_embedding_model_source_radio(embedding_model_source_radio, hub_embedding_model_names_dropdown):
        if embedding_model_source_radio == "Download From Huggingface Hub":
            if not hub_embedding_model_names_dropdown:
                model_download_status = '<span style="color:red">&nbsp;&nbsp;&nbsp;&nbsp;No model is selected.</span>'
            else:
                model_config_path = os.path.join(local_embedding_model_dir, hub_embedding_model_names_dropdown,
                                                 "config.json")
                model_config_path1 = os.path.join(local_embedding_model_dir, hub_embedding_model_names_dropdown,
                                                  "pytorch_model.bin")
                model_config_path2 = os.path.join(local_embedding_model_dir, hub_embedding_model_names_dropdown,
                                                  "model.safetensors")

                if os.path.exists(model_config_path):
                    model_download_status = '<span style="color:green">&nbsp;&nbsp;&nbsp;&nbsp;This model has already been downloaded to local.</span>'
                else:
                    model_download_status = '<span style="color:red">&nbsp;&nbsp;&nbsp;&nbsp;This model has not been downloaded.</span>'
            return gr.update(visible=True), gr.update(visible=False), gr.update(
                visible=False), gr.update(visible=True, value=model_download_status), gr.update(
                visible=True), gr.update(visible=False)
        else:
            model_download_status = ""
            return gr.update(visible=False), gr.update(visible=True), gr.update(
                visible=True), gr.update(visible=False, value=model_download_status), gr.update(
                visible=False), gr.update(visible=False)


    click_download_hub_embedding_model_names_btn_event = download_hub_embedding_model_names_btn.click(
        check_local_model_or_dataset_is_empty4, [hub_embedding_model_names_dropdown,Huggingface_hub_token]).success(
        click_download_hub_embedding_model_btn, [],
        [download_hub_embedding_model_names_btn,
         stop_download_hub_embedding_model_names_btn,
         download_hub_embedding_model_status_markdown]).then(
        download_model_wrapper, [hub_embedding_model_names_dropdown, local_embedding_model_root_dir_textbox],
        download_hub_embedding_model_status_markdown). \
        then(download_hub_embedding_model_postprocess, [],
             [download_hub_embedding_model_names_btn, stop_download_hub_embedding_model_names_btn])

    stop_download_hub_embedding_model_names_btn.click(click_stop_download_hub_embedding_model_names_btn, [],
                                                      [download_hub_embedding_model_names_btn,
                                                       stop_download_hub_embedding_model_names_btn,
                                                       download_hub_embedding_model_status_markdown],
                                                      cancels=[click_download_hub_embedding_model_names_btn_event])
    embedding_model_source_radio.change(change_embedding_model_source_radio,
                                        [embedding_model_source_radio, hub_embedding_model_names_dropdown],
                                        [hub_embedding_model_names_dropdown, local_embedding_model_names_dropdown,
                                         refresh_local_embedding_model_names_btn,
                                         download_hub_embedding_model_status_markdown,
                                         download_hub_embedding_model_names_btn,
                                         stop_download_hub_embedding_model_names_btn],
                                        cancels=[click_download_hub_embedding_model_names_btn_event])
    def click_refresh_local_embedding_model_names_btn():
        local_embedding_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag", "embedding_models")
        local_embedding_model_names = get_hg_model_names_from_dir(local_embedding_model_dir, "embedding_models")
        return gr.update(choices=local_embedding_model_names,
                         value=local_embedding_model_names[0] if local_embedding_model_names else None)
    refresh_local_embedding_model_names_btn.click(click_refresh_local_embedding_model_names_btn,[],local_embedding_model_names_dropdown)
    def change_hub_embedding_model_names_dropdown(hub_embedding_model_names_dropdown):
        if hub_embedding_model_names_dropdown == "None":
            return gr.update(visible=True,
                             value='<span style="color:red">&nbsp;&nbsp;&nbsp;&nbsp;No model is selected.</span>'), \
                   gr.update(visible=True), gr.update(visible=False)

        model_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag", "embedding_models",
                                         hub_embedding_model_names_dropdown, "config.json")
        if os.path.exists(model_config_path):
            return gr.update(
                visible=True,
                value='<span style="color:green">&nbsp;&nbsp;&nbsp;&nbsp;This model has already been downloaded to local.</span>'), \
                   gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=True,
                             value='<span style="color:red">&nbsp;&nbsp;&nbsp;&nbsp;This model has not been downloaded.</span>'), \
                   gr.update(visible=True), gr.update(visible=False)


    hub_embedding_model_names_dropdown.change(change_hub_embedding_model_names_dropdown,
                                              hub_embedding_model_names_dropdown,
                                              [download_hub_embedding_model_status_markdown,
                                               download_hub_embedding_model_names_btn,
                                               stop_download_hub_embedding_model_names_btn],
                                              cancels=[click_download_hub_embedding_model_names_btn_event])


    def download_hub_chat_model_postprocess():
        return gr.update(visible=True), gr.update(visible=False)


    def click_download_hub_chat_model_btn():
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)


    def click_stop_download_hub_chat_model_names_btn():
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)


    def click_stop_download_hub_chat_model_names_btn():
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)


    def change_chat_model_source_radio(chat_model_source_radio, hub_chat_model_names_dropdown):
        local_chat_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        if chat_model_source_radio == "Download From Huggingface Hub":
            if not hub_chat_model_names_dropdown:
                model_download_status = '<span style="color:red">&nbsp;&nbsp;&nbsp;&nbsp;No model is selected.</span>'
            else:
                if validate_model_path(hub_chat_model_names_dropdown)[0]:
                    model_download_status = '<span style="color:green">&nbsp;&nbsp;&nbsp;&nbsp;This model has already been downloaded to local.</span>'
                else:
                    model_download_status = '<span style="color:red">&nbsp;&nbsp;&nbsp;&nbsp;This model has not been downloaded.</span>'
            return gr.update(visible=True), gr.update(visible=False), gr.update(
                visible=False), gr.update(visible=True, value=model_download_status), gr.update(
                visible=True), gr.update(visible=False)
        else:
            model_download_status = ""
            return gr.update(visible=False), gr.update(visible=True), gr.update(
                visible=True), gr.update(visible=False, value=model_download_status), gr.update(
                visible=False), gr.update(visible=False)


    click_download_hub_chat_model_names_btn_event = download_hub_chat_model_names_btn.click(
        check_local_model_or_dataset_is_empty5, [hub_chat_model_names_dropdown,Huggingface_hub_token]).success(
        click_download_hub_chat_model_btn, [],
        [download_hub_chat_model_names_btn,
         stop_download_hub_chat_model_names_btn,
         download_hub_chat_model_status_markdown]).then(
        download_model_wrapper, [hub_chat_model_names_dropdown, local_chat_model_root_dir_textbox],
        download_hub_chat_model_status_markdown). \
        then(download_hub_chat_model_postprocess, [],
             [download_hub_chat_model_names_btn, stop_download_hub_chat_model_names_btn])

    stop_download_hub_chat_model_names_btn.click(click_stop_download_hub_chat_model_names_btn, [],
                                                 [download_hub_chat_model_names_btn,
                                                  stop_download_hub_chat_model_names_btn,
                                                  download_hub_chat_model_status_markdown],
                                                 cancels=[click_download_hub_chat_model_names_btn_event])
    chat_model_source_radio.change(change_chat_model_source_radio,
                                   [chat_model_source_radio, hub_chat_model_names_dropdown],
                                   [hub_chat_model_names_dropdown, local_chat_model_names_dropdown,
                                    refresh_local_chat_model_names_btn, download_hub_chat_model_status_markdown,
                                    download_hub_chat_model_names_btn, stop_download_hub_chat_model_names_btn],
                                   cancels=[click_download_hub_chat_model_names_btn_event])

    def click_refresh_local_chat_model_names_btn():
        local_chat_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        runs_model_root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
        # local_chat_model_names = get_hg_model_names_from_dir(local_chat_model_dir)
        local_chat_model_names = get_hg_model_names_and_gguf_from_dir(local_chat_model_dir, runs_model_root_dir)
        return gr.update(choices=local_chat_model_names,
                         value=local_chat_model_names[0] if local_chat_model_names else None)
    refresh_local_chat_model_names_btn.click(click_refresh_local_chat_model_names_btn,[],local_chat_model_names_dropdown)
    def change_hub_chat_model_names_dropdown(hub_chat_model_names_dropdown):
        if hub_chat_model_names_dropdown == "None":
            return gr.update(visible=True,
                             value='<span style="color:red">&nbsp;&nbsp;&nbsp;&nbsp;No model is selected.</span>'), \
                   gr.update(visible=True), gr.update(visible=False)
        if validate_model_path(hub_chat_model_names_dropdown)[0]:
            return gr.update(
                visible=True,
                value='<span style="color:green">&nbsp;&nbsp;&nbsp;&nbsp;This model has already been downloaded to local.</span>'), \
                   gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=True,
                             value='<span style="color:red">&nbsp;&nbsp;&nbsp;&nbsp;This model has not been downloaded.</span>'), \
                   gr.update(visible=True), gr.update(visible=False)


    hub_chat_model_names_dropdown.change(change_hub_chat_model_names_dropdown, hub_chat_model_names_dropdown,
                                         [download_hub_chat_model_status_markdown,
                                          download_hub_chat_model_names_btn, stop_download_hub_chat_model_names_btn],
                                         cancels=[click_download_hub_chat_model_names_btn_event])


    def rag_click_stop_btn():
        global stop_generation_status
        stop_generation_status = True

    def rag_clear_chat_history():
        global chatbot_history, stop_generation_status
        stop_generation_status = True
        chatbot_history = []
        return gr.update(value=None)

    def rag_show_chatbot_question1(text):
        global chatbot_history
        if not text:
            raise gr.Error('Enter text')
        chatbot_history = chatbot_history + [[text, '']]
        chatbot_history = chatbot_history[-2:]
        return chatbot_history
    def rag_show_chatbot_question2(text):
        global chatbot_history
        if not text:
            raise gr.Error('Enter text')
        chatbot_history = chatbot_history + [[text, '']]
        chatbot_history = chatbot_history[-2:]
        return chatbot_history

    # input_txtbox.submit(add_text)
    def rag_chat_with_model1(input_txtbox):
        global chatbot_history, infer_model, stop_generation_status
        global qa_with_rag
        stop_generation_status = False
        answer, retrieved_txt_list = qa_with_rag.generate(input_txtbox)
        if not answer:
            raise gr.Error(f"Generating error:{retrieved_txt_list}")

        cid_list = []
        for cid in range(1, len(retrieved_txt_list) + 1):
            cid_list.append(str(cid))
        print("answer:",answer)
        for char in answer:
            if stop_generation_status:
                print("stop_generation_status:", stop_generation_status)
                break
            try:
                chatbot_history[-1][-1] += char
            except:
                break
            time.sleep(0.05)
            yield chatbot_history, gr.update(), f"",gr.update(value="")

        yield chatbot_history, gr.update(
            value=list(zip(cid_list, retrieved_txt_list))) if retrieved_txt_list else gr.update(
            value=None), f"### &nbsp;&nbsp;Retrieved Document Chunks({len(retrieved_txt_list)} items)",gr.update(value="")
    def rag_chat_with_model2(input_txtbox):
        global chatbot_history, infer_model, stop_generation_status
        global qa_with_rag
        stop_generation_status = False
        answer, retrieved_txt_list = qa_with_rag.generate(input_txtbox)
        if not answer:
            raise gr.Error(f"Generating error:{retrieved_txt_list}")

        cid_list = []
        for cid in range(1, len(retrieved_txt_list) + 1):
            cid_list.append(str(cid))
        for char in answer:
            if stop_generation_status:
                break
            try:
                chatbot_history[-1][-1] += char
            except:
                break
            time.sleep(0.05)
            yield chatbot_history, gr.update(), f"",gr.update(value="")

        yield chatbot_history, gr.update(
            value=list(zip(cid_list, retrieved_txt_list))) if retrieved_txt_list else gr.update(
            value=None), f"### &nbsp;&nbsp;Retrieved Document Chunks({len(retrieved_txt_list)} items)",gr.update(value="")


    def check_rag_config1(rag_input_txtbox, rag_data_list_dropdown, text_splitter_dropdown, chunk_size_slider,
                         chunk_overlap_slider,
                         Separators_textbox,
                         embedding_model_source_radio, hub_embedding_model_names_dropdown,
                         local_embedding_model_names_dropdown, chat_model_source_radio,
                         hub_chat_model_names_dropdown, local_chat_model_names_dropdown,
                         search_top_k_slider, search_score_threshold_slider,rag_using_4bit_quantization_checkbox,low_cpu_mem_usage_checkbox,
                         max_new_tokens_slider, temperature_slider, top_k_slider, top_p_slider, repeat_penalty_slider
                         ):
        global RAG_DATA_LIST_DROPDOWN, TEXT_SPLITTER_DROPDOWN, CHUNK_SIZE_SLIDER, CHUNK_OVERLAP_SLIDER, SEPARATORS_TEXTBOX
        global EMBEDDING_MODEL_SOURCE_RADIO, HUB_EMBEDDING_MODEL_NAMES_DROPDOWN, LOCAL_EMBEDDING_MODEL_NAMES_DROPDOWN
        global CHAT_MODEL_SOURCE_RADIO, HUB_CHAT_MODEL_NAMES_DROPDOWN, LOCAL_CHAT_MODEL_NAMES_DROPDOWN
        global qa_with_rag, SEARCH_TOP_K_SLIDER, SEARCH_SCORE_THRESHOLD_SLIDER

        if TRAINING_STATUS.status == 0:
            raise gr.Error("Training is running,you should stop training before start chatting!")

        if not rag_input_txtbox.strip():
            raise gr.Error("Input text is empty!")

        config_status = {"text_splitter": False,
                         "embedding_model": False,
                         "chat_model": False
                         }


        if chat_model_source_radio == "Download From Huggingface Hub":
            if HUB_CHAT_MODEL_NAMES_DROPDOWN != hub_chat_model_names_dropdown:
                if not hub_chat_model_names_dropdown:
                    yield ""
                    raise gr.Error("Chat model is empty!")
                if not validate_model_path(hub_chat_model_names_dropdown)[0]:
                    yield ""
                    raise gr.Error("Chat model does not exist!")
                    return ""
                # yield "<span style='color:green'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Loading chat model ... </span>"
                yield "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Loading chat model ... "
                if qa_with_rag:
                  qa_with_rag.free_memory()
                  # del qa_with_rag
                  # qa_with_rag = None
                # qa_with_rag = QAWithRAG()
                gc.collect()
                with torch.no_grad():
                  torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                gc.collect()
                model_path = validate_model_path(hub_chat_model_names_dropdown)[1]
                load_model_status, msg = qa_with_rag.load_chat_model(model_path,rag_using_4bit_quantization_checkbox,low_cpu_mem_usage_checkbox,
                                                                     max_new_tokens_slider, temperature_slider,
                                                                     top_k_slider, top_p_slider, repeat_penalty_slider
                                                                     )
                if load_model_status == -1:
                    raise gr.Error(f"Loading chat model error:{msg}")

                HUB_CHAT_MODEL_NAMES_DROPDOWN = hub_chat_model_names_dropdown
                config_status["chat_model"] = True
        else:
            if LOCAL_CHAT_MODEL_NAMES_DROPDOWN != local_chat_model_names_dropdown:
                if not local_chat_model_names_dropdown:
                    yield ""
                    raise gr.Error("Chat model is empty!")
                # yield "<span style='color:green'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Loading chat model ... </span>"
                yield "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Loading chat model ..."
                if qa_with_rag:
                  qa_with_rag.free_memory()
                  # del qa_with_rag
                  # qa_with_rag = None
                # qa_with_rag = QAWithRAG()
                gc.collect()
                with torch.no_grad():
                  torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                gc.collect()

                model_path = validate_model_path(hub_chat_model_names_dropdown)[1]
                load_model_status, msg = qa_with_rag.load_chat_model(model_path,rag_using_4bit_quantization_checkbox,
                                                                     low_cpu_mem_usage_checkbox,
                    max_new_tokens_slider, temperature_slider, top_k_slider, top_p_slider, repeat_penalty_slider
                )
                if load_model_status == -1:
                    raise gr.Error(f"Loading chat model error:{msg}")
                LOCAL_CHAT_MODEL_NAMES_DROPDOWN = local_chat_model_names_dropdown
                config_status["chat_model"] = True

        if embedding_model_source_radio == "Download From Huggingface Hub":
            if HUB_EMBEDDING_MODEL_NAMES_DROPDOWN != hub_embedding_model_names_dropdown:
                if not hub_embedding_model_names_dropdown:
                    yield ""
                    raise gr.Error("Embedding model is empty!")
                config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag", "embedding_models",
                                           hub_embedding_model_names_dropdown, "config.json")
                if not os.path.exists(config_path):
                    yield ""
                    raise gr.Error("Embedding model does not exist!")
                yield "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Loading embedding model ..."
                model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag", "embedding_models",
                                          hub_embedding_model_names_dropdown)
                if qa_with_rag.embedding_function:
                    del qa_with_rag.embedding_function
                    qa_with_rag.embedding_function = None
                try:
                    qa_with_rag.load_embedding_model(model_path)
                except Exception as e:
                    raise gr.Error(f"Loading embedding model error:{e}")
                HUB_EMBEDDING_MODEL_NAMES_DROPDOWN = hub_embedding_model_names_dropdown
                config_status["embedding_model"] = True
        else:
            if LOCAL_EMBEDDING_MODEL_NAMES_DROPDOWN != local_embedding_model_names_dropdown:
                if not local_embedding_model_names_dropdown:
                    yield ""
                    raise gr.Error("Embedding model is empty!")
                yield "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Loading embedding model..."
                local_embedding_model_dir = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "rag", "embedding_models")
                try:
                    qa_with_rag.load_embedding_model(
                        os.path.join(local_embedding_model_dir, local_embedding_model_names_dropdown))
                except Exception as e:
                    raise gr.Error(f"Loading embedding model error:{e}")
                LOCAL_EMBEDDING_MODEL_NAMES_DROPDOWN = local_embedding_model_names_dropdown
                config_status["embedding_model"] = True

        bool_value_list = []
        bool_value_list.append(TEXT_SPLITTER_DROPDOWN != text_splitter_dropdown)
        bool_value_list.append(CHUNK_SIZE_SLIDER != chunk_size_slider)
        bool_value_list.append(CHUNK_OVERLAP_SLIDER != chunk_overlap_slider)
        bool_value_list.append(SEPARATORS_TEXTBOX != Separators_textbox)
        bool_value_list = np.asarray(bool_value_list)

        if np.any(bool_value_list[0:4]):
            try:
                qa_with_rag.get_text_splitter(chunk_size_slider, chunk_overlap_slider,
                                              Separators_textbox)
            except Exception as e:
                raise gr.Error(f"Initializing text splitter error:{e}")
            yield "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Initializing text splitter ..."
            TEXT_SPLITTER_DROPDOWN = text_splitter_dropdown
            CHUNK_SIZE_SLIDER = chunk_size_slider
            CHUNK_OVERLAP_SLIDER = chunk_overlap_slider
            SEPARATORS_TEXTBOX = Separators_textbox
            config_status["text_splitter"] = True

        if RAG_DATA_LIST_DROPDOWN != rag_data_list_dropdown or config_status["text_splitter"] or \
                config_status["embedding_model"]:
            yield "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Generating vector store ... "
            # try:
            #     qa_with_rag.add_document_to_vector_store(
            #         os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag", "data",
            #                      rag_data_list_dropdown), search_top_k_slider, search_score_threshold_slider)
            # except Exception as e:
            #     raise gr.Error(f"Generating vector store error:{e}")
            qa_with_rag.add_document_to_vector_store(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag", "data",
                              rag_data_list_dropdown), search_top_k_slider, search_score_threshold_slider)
            RAG_DATA_LIST_DROPDOWN = rag_data_list_dropdown
            yield ""
        if SEARCH_TOP_K_SLIDER != search_top_k_slider or SEARCH_SCORE_THRESHOLD_SLIDER != search_score_threshold_slider:
            yield "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Setting retriever ... "
            try:
                qa_with_rag.set_retriever(search_top_k_slider, search_score_threshold_slider)
            except Exception as e:
                raise gr.Error(f"Setting retriever error:{e}")
            SEARCH_SCORE_THRESHOLD_SLIDER = search_score_threshold_slider
            SEARCH_TOP_K_SLIDER = search_top_k_slider
            yield ""
        yield ""
    def check_rag_config2(rag_input_txtbox, rag_data_list_dropdown, text_splitter_dropdown, chunk_size_slider,
                         chunk_overlap_slider,
                         Separators_textbox,
                         embedding_model_source_radio, hub_embedding_model_names_dropdown,
                         local_embedding_model_names_dropdown, chat_model_source_radio,
                         hub_chat_model_names_dropdown, local_chat_model_names_dropdown,
                         search_top_k_slider, search_score_threshold_slider,rag_using_4bit_quantization_checkbox,low_cpu_mem_usage_checkbox,
                         max_new_tokens_slider, temperature_slider, top_k_slider, top_p_slider, repeat_penalty_slider
                         ):
        global RAG_DATA_LIST_DROPDOWN, TEXT_SPLITTER_DROPDOWN, CHUNK_SIZE_SLIDER, CHUNK_OVERLAP_SLIDER, SEPARATORS_TEXTBOX
        global EMBEDDING_MODEL_SOURCE_RADIO, HUB_EMBEDDING_MODEL_NAMES_DROPDOWN, LOCAL_EMBEDDING_MODEL_NAMES_DROPDOWN
        global CHAT_MODEL_SOURCE_RADIO, HUB_CHAT_MODEL_NAMES_DROPDOWN, LOCAL_CHAT_MODEL_NAMES_DROPDOWN
        global qa_with_rag, SEARCH_TOP_K_SLIDER, SEARCH_SCORE_THRESHOLD_SLIDER

        if TRAINING_STATUS.status == 0:
            raise gr.Error("Training is running,you should stop training before start chatting!")

        if not rag_input_txtbox.strip():
            raise gr.Error("Input text is empty!")

        config_status = {"text_splitter": False,
                         "embedding_model": False,
                         "chat_model": False
                         }


        if chat_model_source_radio == "Download From Huggingface Hub":
            if HUB_CHAT_MODEL_NAMES_DROPDOWN != hub_chat_model_names_dropdown:
                if not hub_chat_model_names_dropdown:
                    yield ""
                    raise gr.Error("Chat model is empty!")
                if not validate_model_path(hub_chat_model_names_dropdown)[0]:
                    yield ""
                    raise gr.Error("Chat model does not exist!")
                    return ""
                # yield "<span style='color:green'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Loading chat model ... </span>"
                yield "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Loading chat model ... "
                if qa_with_rag:
                  qa_with_rag.free_memory()
                  # del qa_with_rag
                  # qa_with_rag = None
                # qa_with_rag = QAWithRAG()
                gc.collect()
                with torch.no_grad():
                  torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                gc.collect()
                model_path = validate_model_path(hub_chat_model_names_dropdown)[1]
                load_model_status, msg = qa_with_rag.load_chat_model(model_path,rag_using_4bit_quantization_checkbox,
                                                                     low_cpu_mem_usage_checkbox,
                                                                     max_new_tokens_slider, temperature_slider,
                                                                     top_k_slider, top_p_slider, repeat_penalty_slider
                                                                     )
                if load_model_status == -1:
                    raise gr.Error(f"Loading chat model error:{msg}")

                HUB_CHAT_MODEL_NAMES_DROPDOWN = hub_chat_model_names_dropdown
                config_status["chat_model"] = True
        else:
            if LOCAL_CHAT_MODEL_NAMES_DROPDOWN != local_chat_model_names_dropdown:
                if not local_chat_model_names_dropdown:
                    yield ""
                    raise gr.Error("Chat model is empty!")
                # yield "<span style='color:green'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Loading chat model ... </span>"
                yield "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Loading chat model ..."
                if qa_with_rag:
                  qa_with_rag.free_memory()
                  # del qa_with_rag
                  # qa_with_rag = None
                # qa_with_rag = QAWithRAG()
                gc.collect()
                with torch.no_grad():
                  torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                gc.collect()
                model_path = validate_model_path(hub_chat_model_names_dropdown)[1]
                load_model_status, msg = qa_with_rag.load_chat_model(
                    model_path, rag_using_4bit_quantization_checkbox,
                    low_cpu_mem_usage_checkbox,
                    max_new_tokens_slider, temperature_slider, top_k_slider, top_p_slider, repeat_penalty_slider
                )
                if load_model_status == -1:
                    raise gr.Error(f"Loading chat model error:{msg}")
                LOCAL_CHAT_MODEL_NAMES_DROPDOWN = local_chat_model_names_dropdown
                config_status["chat_model"] = True

        if embedding_model_source_radio == "Download From Huggingface Hub":
            if HUB_EMBEDDING_MODEL_NAMES_DROPDOWN != hub_embedding_model_names_dropdown:
                if not hub_embedding_model_names_dropdown:
                    yield ""
                    raise gr.Error("Embedding model is empty!")
                config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag", "embedding_models",
                                           hub_embedding_model_names_dropdown, "config.json")
                if not os.path.exists(config_path):
                    yield ""
                    raise gr.Error("Embedding model does not exist!")
                yield "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Loading embedding model ..."
                model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag", "embedding_models",
                                          hub_embedding_model_names_dropdown)
                if qa_with_rag.embedding_function:
                    del qa_with_rag.embedding_function
                    qa_with_rag.embedding_function = None
                try:
                    qa_with_rag.load_embedding_model(model_path)
                except Exception as e:
                    raise gr.Error(f"Loading embedding model error:{e}")
                HUB_EMBEDDING_MODEL_NAMES_DROPDOWN = hub_embedding_model_names_dropdown
                config_status["embedding_model"] = True
        else:
            if LOCAL_EMBEDDING_MODEL_NAMES_DROPDOWN != local_embedding_model_names_dropdown:
                if not local_embedding_model_names_dropdown:
                    yield ""
                    raise gr.Error("Embedding model is empty!")
                yield "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Loading embedding model..."
                local_embedding_model_dir = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "rag", "embedding_models")
                try:
                    qa_with_rag.load_embedding_model(
                        os.path.join(local_embedding_model_dir, local_embedding_model_names_dropdown))
                except Exception as e:
                    raise gr.Error(f"Loading embedding model error:{e}")
                LOCAL_EMBEDDING_MODEL_NAMES_DROPDOWN = local_embedding_model_names_dropdown
                config_status["embedding_model"] = True

        bool_value_list = []
        bool_value_list.append(TEXT_SPLITTER_DROPDOWN != text_splitter_dropdown)
        bool_value_list.append(CHUNK_SIZE_SLIDER != chunk_size_slider)
        bool_value_list.append(CHUNK_OVERLAP_SLIDER != chunk_overlap_slider)
        bool_value_list.append(SEPARATORS_TEXTBOX != Separators_textbox)
        bool_value_list = np.asarray(bool_value_list)

        if np.any(bool_value_list[0:4]):
            try:
                qa_with_rag.get_text_splitter(chunk_size_slider, chunk_overlap_slider,
                                              Separators_textbox)
            except Exception as e:
                raise gr.Error(f"Initializing text splitter error:{e}")
            yield "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Initializing text splitter ..."
            TEXT_SPLITTER_DROPDOWN = text_splitter_dropdown
            CHUNK_SIZE_SLIDER = chunk_size_slider
            CHUNK_OVERLAP_SLIDER = chunk_overlap_slider
            SEPARATORS_TEXTBOX = Separators_textbox
            config_status["text_splitter"] = True

        if RAG_DATA_LIST_DROPDOWN != rag_data_list_dropdown or config_status["text_splitter"] or \
                config_status["embedding_model"]:
            yield "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Generating vector store ... "
            # try:
            #     qa_with_rag.add_document_to_vector_store(
            #         os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag", "data",
            #                      rag_data_list_dropdown), search_top_k_slider, search_score_threshold_slider)
            # except Exception as e:
            #     raise gr.Error(f"Generating vector store error:{e}")
            qa_with_rag.add_document_to_vector_store(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag", "data",
                              rag_data_list_dropdown), search_top_k_slider, search_score_threshold_slider)
            RAG_DATA_LIST_DROPDOWN = rag_data_list_dropdown
            yield ""
        if SEARCH_TOP_K_SLIDER != search_top_k_slider or SEARCH_SCORE_THRESHOLD_SLIDER != search_score_threshold_slider:
            yield "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Setting retriever ... "
            try:
                qa_with_rag.set_retriever(search_top_k_slider, search_score_threshold_slider)
            except Exception as e:
                raise gr.Error(f"Setting retriever error:{e}")
            SEARCH_SCORE_THRESHOLD_SLIDER = search_score_threshold_slider
            SEARCH_TOP_K_SLIDER = search_top_k_slider
            yield ""
        yield ""

    check_rag_config_params = [rag_input_txtbox, rag_data_list_dropdown, text_splitter_dropdown,
                               chunk_size_slider, chunk_overlap_slider,
                               Separators_textbox,
                               embedding_model_source_radio, hub_embedding_model_names_dropdown,
                               local_embedding_model_names_dropdown, chat_model_source_radio,
                               hub_chat_model_names_dropdown, local_chat_model_names_dropdown,
                               search_top_k_slider, search_score_threshold_slider,rag_using_4bit_quantization_checkbox,
                               low_cpu_mem_usage_checkbox,
                               max_new_tokens_slider, temperature_slider, top_k_slider, top_p_slider,
                               repeat_penalty_slider
                               ]
    def rag_generate_btn_click_clear_text1():
        return gr.update(value="")
    def rag_generate_btn_click_clear_text2():
        return gr.update(value="")


    rag_input_txtbox.submit(check_rag_config1, check_rag_config_params, rag_model_running_status_markdown). \
        success(rag_show_chatbot_question1, inputs=[rag_input_txtbox], outputs=[rag_chatbot], queue=False). \
        success(rag_chat_with_model1, inputs=[rag_input_txtbox],
                outputs=[rag_chatbot, retrieved_document_chunks_dataframe, rag_model_running_status_markdown,rag_input_txtbox])
    rag_generate_btn.click(check_rag_config2, check_rag_config_params, rag_model_running_status_markdown). \
        success(rag_show_chatbot_question2, inputs=[rag_input_txtbox], outputs=[rag_chatbot], queue=False). \
        success(rag_chat_with_model2, inputs=[rag_input_txtbox],
                outputs=[rag_chatbot, retrieved_document_chunks_dataframe, rag_model_running_status_markdown,rag_input_txtbox])
    # rag_clear_btn.click(rag_clear_chat_history, [], rag_chatbot)
    rag_stop_btn.click(rag_click_stop_btn)

demo.queue(concurrency_count=3)
demo.launch(share=True)
# demo.launch(False=True,server_name="0.0.0.0")