import yaml
import os
import datasets
import importlib
import gradio as gr
import glob
from dotenv import load_dotenv
from huggingface_hub import login

def login_huggingface(token,base_model_name_dropdown):
    if base_model_name_dropdown.lower().find("llama") >= 0:
        if token:
            HUGGINGFACE_HUB_TOKEN = token
            print("d1:",HUGGINGFACE_HUB_TOKEN)
        else:
            env_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"token.env")
            load_dotenv(env_file_path)
            HUGGINGFACE_HUB_TOKEN = os.getenv('HUGGINGFACE_HUB_TOKEN')
            print("d2:", HUGGINGFACE_HUB_TOKEN)
        login(token=HUGGINGFACE_HUB_TOKEN)
        os.environ["HUGGING_FACE_HUB_TOKEN"] = HUGGINGFACE_HUB_TOKEN


def get_runs_models():
    training_runs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'runs')
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
                    runs_output_model.append(os.path.join(run_name, "output_model", run_output_model_name, "ori"))
    runs_output_model = runs_output_model[::-1]
    return runs_output_model

def get_runs_model_names_from_dir(root_dir):

    run_names = os.listdir(root_dir)
    run_names.sort(key=lambda file: os.path.getmtime(os.path.join(root_dir, file)),reverse=True)
    runs_output_model = []
    for run_name in run_names:
        run_name_dir = os.path.join(root_dir, run_name)
        run_output_model = os.path.join(run_name_dir, "output_model")
        if os.path.exists(run_output_model):
            run_output_model_names = os.listdir(run_output_model)
            for run_output_model_name in run_output_model_names:
                model_bin_path = os.path.exists(
                    os.path.join(root_dir,
                                 run_name, "output_model", run_output_model_name, "ori",
                                 "pytorch_model.bin"))
                if run_output_model_name.find("merged_") >= 0 and model_bin_path:
                    runs_output_model.append(os.path.join(run_name, "output_model", run_output_model_name, "ori"))
    return runs_output_model

def validate_model_path(model_name):
    if not model_name:
        return False,""
    home_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    base_model_config_path1 = os.path.join(home_dir, "models", model_name)
    base_model_config_path2 = os.path.join(base_model_config_path1, "config.json")
    run_model_config_path1 = os.path.join(home_dir, "runs", model_name)
    run_model_config_path2 = os.path.join(run_model_config_path1, "config.json")
    if os.path.exists(base_model_config_path1) and base_model_config_path1.endswith(".gguf"):
        return True,base_model_config_path1
    if os.path.exists(run_model_config_path1) and run_model_config_path1.endswith(".gguf") :
        return True,run_model_config_path1
    if os.path.exists(base_model_config_path2):
        return True,base_model_config_path1
    if os.path.exists(run_model_config_path2):
        return True,run_model_config_path1
    return False,""
def get_hg_model_names_from_dir(root_dir):
    model_names = os.listdir(root_dir)
    model_names.sort(key=lambda file: os.path.getmtime(os.path.join(root_dir, file)),reverse=True)
    return model_names
def get_hg_model_names_from_dir(root_dir,prefix="models"):
    output = []
    model_names_1 = glob.glob(os.path.join(root_dir, "**", "**", "config.json"), recursive=False)
    model_names_2 = glob.glob(os.path.join(root_dir, "**","config.json"), recursive=False)
    model_names_1 += model_names_2
    for name in model_names_1:
        output.append(name[name.find(prefix)+len(prefix)+1:name.find("config.json")-1])
    return output

def get_hg_model_names_and_gguf_from_dir(hg_model_root_dir,runs_model_root_dir):
    output = []
    runs_gguf_files = glob.glob(os.path.join(runs_model_root_dir,"**","**","**","**","*.gguf"),recursive=False)
    root_model_gguf_files = glob.glob(os.path.join(hg_model_root_dir,"**","*.gguf"),recursive=False)
    root_model_gguf_files1 = glob.glob(os.path.join(hg_model_root_dir, "**","**",  "*.gguf"), recursive=False)
    root_model_hg_dir0 = glob.glob(os.path.join(hg_model_root_dir,"**","config.json"),recursive=False)
    root_model_hg_dir1 = glob.glob(os.path.join(hg_model_root_dir, "**","**",  "config.json"), recursive=False)
    runs_hg_dir = glob.glob(os.path.join(hg_model_root_dir,"**","**","**","**","config.json"),recursive=False)
    runs_gguf_files.sort(key=lambda file: os.path.getmtime(file), reverse=True)
    root_model_gguf_files.sort(key=lambda file: os.path.getmtime(file), reverse=True)
    root_model_gguf_files1.sort(key=lambda file: os.path.getmtime(file), reverse=True)
    root_model_hg_dir0.sort(key=lambda file: os.path.getmtime(file), reverse=True)
    root_model_hg_dir1.sort(key=lambda file: os.path.getmtime(file), reverse=True)
    runs_hg_dir.sort(key=lambda file: os.path.getmtime(file), reverse=True)

    for file in runs_gguf_files:
        file_pos = file.find("runs")
        output.append(file[file_pos:])
    for file in root_model_gguf_files:
        output.append(file[file.find("models")+len("models")+1:])
    for file in root_model_gguf_files1:
        output.append(file[file.find("models")+len("models")+1:])
    for file in root_model_hg_dir0:
        file_pos1 = file.find("models")
        file_pos2 = file.find("config.json")
        output.append(file[file_pos1+len("models")+1:file_pos2-1])
    for file in root_model_hg_dir1:
        file_pos1 = file.find("models")
        file_pos2 = file.find("config.json")
        output.append(file[file_pos1+len("models")+1:file_pos2-1])
    for file in runs_hg_dir:
        file_pos = file.find("runs")+len("runs")+1
        output.append(file[file_pos:])
    return output

def read_yaml(yaml_path):
    with open(yaml_path) as f1:
        try:
            data = yaml.safe_load(f1)
            return data
        except yaml.YAMLError as e:
            raise ValueError(f'Error loading yaml file: {e}')

def get_first_row_from_dataset(dataset_path):
    if os.path.exists(os.path.join(dataset_path, "dataset_dict.json")):
        dataset = datasets.load_from_disk(dataset_path)
    elif os.path.exists(os.path.join(dataset_path, "dataset_infos.json")):
        dataset = datasets.load_dataset(dataset_path)
    elif os.path.exists(os.path.join(dataset_path, "dataset_info.json")):
        dataset = datasets.load_from_disk(dataset_path)
    else:
        raise ValueError(
            f'Invalid Dataset format {dataset_path}.')
    try:
        split_list = list(dataset.keys())
    except:
        split_list = ["train"]
    new_split_list= ["","",""]
    for split in split_list:
        if split.find("train") >= 0:
            new_split_list[0] = split
        elif split.find("val") >= 0:
            new_split_list[1] = split
        elif split.find("test") >= 0:
            new_split_list[2] = split

    return dataset[new_split_list[0]][0],new_split_list