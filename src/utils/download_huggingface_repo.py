
from datasets import load_dataset
import os
from pathlib import Path
import numpy as np
from src.utils import  download_model
from huggingface_hub import hf_hub_download
import traceback
import gradio as gr
def download_model_wrapper(repo_id,local_model_root_dir, specific_file=None, return_links=False, check=False,progress = gr.Progress()):
    if repo_id.endswith(".gguf"):
        try:
            model_dir = os.path.join(local_model_root_dir, '/'.join(repo_id.split('/')[0:-1]))
            yield f"<span style='color:green'>&nbsp;&nbsp;&nbsp;&nbsp;Downloading file {repo_id.split('/')[-1]} to `{model_dir}/...`</span>"
            hf_hub_download(repo_id='/'.join(repo_id.split('/')[0:-1]), filename=repo_id.split('/')[-1], local_dir=model_dir, resume_download=True,
                            force_download=False)
        except:
            progress(1.0)
            yield traceback.format_exc().replace('\n', '\n\n')
        yield "<span style='color:green'>&nbsp;&nbsp;&nbsp;&nbsp;Download successful!</span>"
    else:
        if repo_id == "" or repo_id == "None":
            # return gr.update(value="Model's name is empty!",visible=True)
            yield f"Model's name is empty!"
        else:
            model_dir = os.path.join(local_model_root_dir, repo_id)

            model_config_path = os.path.join(model_dir, "config.json")
            model_config_path1 = os.path.join(model_dir, "pytorch_model.bin")
            model_config_path2 = os.path.join(model_dir, "model.safetensors")
            if os.path.exists(model_config_path1) or os.path.exists(model_config_path2):
                yield '<span style="color:green">&nbsp;&nbsp;&nbsp;&nbsp;This model has already been downloaded.</span>'
            else:

                try:
                    progress(0.0)
                    # download_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"download-model.py")
                    # downloader = importlib.import_module(download_model_path).ModelDownloader()
                    downloader = download_model.ModelDownloader()
                    model, branch = downloader.sanitize_model_and_branch_names(repo_id, None)
                    yield ("Getting the download links from Hugging Face")
                    links, sha256, is_lora, is_llamacpp, link_file_size_list = downloader.get_download_links_from_huggingface(model,
                                                                                                                              branch,
                                                                                                                              text_only=False,
                                                                                                                              specific_file=specific_file
                                                                                                                              )
                    if return_links:
                        yield '\n\n'.join([f"`{Path(link).name}`" for link in links])
                    yield ("Getting the output folder")
                    # base_folder = shared.args.lora_dir if is_lora else shared.args.model_dir
                    base_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
                    output_folder = downloader.get_output_folder(model, branch, is_lora, is_llamacpp=is_llamacpp,
                                                                 base_folder=base_folder)
                    link_file_size_list = np.array(link_file_size_list)
                    links = np.array(links)
                    sorted_index = np.argsort(link_file_size_list)
                    link_file_size_list = link_file_size_list[sorted_index]
                    links = links[sorted_index]
                    total_file_size = sum(link_file_size_list)
                    copyed_file_size = 0
                    for link, link_file_size in zip(links, link_file_size_list):
                        model_file_name = link.split('/')[-1]
                        if model_file_name.find("Pooling")>=0:
                            model_file_name = model_file_name+"/config.json"
                        # yield (f"Downloading file {model_file_name} to `{output_folder}/...`")
                        yield f"<span style='color:green'>&nbsp;&nbsp;&nbsp;&nbsp;Downloading file {model_file_name} to `{output_folder}/...`</span>"
                        hf_hub_download(repo_id=repo_id, filename=model_file_name, local_dir=model_dir, resume_download=True,
                                        force_download=False)
                        copyed_file_size += link_file_size
                        progress(copyed_file_size / total_file_size)
                    # yield ("Download successful!")
                    yield "<span style='color:green'>&nbsp;&nbsp;&nbsp;&nbsp;Download successful!</span>"
                except:
                    progress(1.0)
                    yield traceback.format_exc().replace('\n', '\n\n')
def download_dataset_wrapper(repo_id,local_dataset_root_dir,progress = gr.Progress()):
    repo_id = repo_id.strip()
    if repo_id == "":
        yield "<span style='color:red'>&nbsp;&nbsp;&nbsp;&nbsp;This Dataset's name is empty!</span>"
    else:
        dataset_dir = os.path.join(local_dataset_root_dir, repo_id)
        # dataset_config_path1 = os.path.join(dataset_dir, "config.json")
        dataset_config_path1 = os.path.join(dataset_dir, "dataset_infos.json")
        dataset_config_path2 = os.path.join(dataset_dir, "dataset_dict.json")

        if os.path.exists(dataset_config_path1) or os.path.exists(dataset_config_path2):
            yield "<span style='color:green'>&nbsp;&nbsp;&nbsp;&nbsp;This Dataset has already been downloaded.</span>"
        else:
            try:

                progress(0.3)
                yield f"<span style='color:green'>&nbsp;&nbsp;&nbsp;&nbsp;Downloading dataset to `{dataset_dir}/...`</span>"
                datasets = load_dataset(repo_id)
                progress(0.8)
                yield "<span style='color:green'>&nbsp;&nbsp;&nbsp;&nbsp;Download successful!</span>"
                datasets.save_to_disk(dataset_dir)
                # datasets = load_from_disk("dddd")
                yield "<span style='color:green'>&nbsp;&nbsp;&nbsp;&nbsp;Download successful!</span>"
            except:
                progress(1.0)
                yield traceback.format_exc().replace('\n', '\n\n')
