'''
Downloads models from Hugging Face to models/username_modelname.

Example:
python download-model.py facebook/opt-1.3b

'''

import argparse
import base64
import datetime
import hashlib
import json
import os
import re
import sys
from pathlib import Path
import numpy as np
import requests
import tqdm
from requests.adapters import HTTPAdapter
from tqdm.contrib.concurrent import thread_map


base = "https://huggingface.co"



class ModelDownloader:
    def __init__(self, max_retries=5):
        self.session = requests.Session()
        if max_retries:
            self.session.mount('https://cdn-lfs.huggingface.co', HTTPAdapter(max_retries=max_retries))
            self.session.mount('https://huggingface.co', HTTPAdapter(max_retries=max_retries))
        # if os.getenv('HF_USER') is not None and os.getenv('HF_PASS') is not None:
            # self.session.auth = (os.getenv('HF_USER'), os.getenv('HF_PASS'))
        if os.getenv('HUGGING_FACE_HUB_TOKEN') is not None:
            self.session.headers = {'authorization': f'Bearer {os.getenv("HUGGING_FACE_HUB_TOKEN")}'}

    def sanitize_model_and_branch_names(self, model, branch):
        if model[-1] == '/':
            model = model[:-1]

        if model.startswith(base + '/'):
            model = model[len(base) + 1:]

        model_parts = model.split(":")
        model = model_parts[0] if len(model_parts) > 0 else model
        branch = model_parts[1] if len(model_parts) > 1 else branch

        if branch is None:
            branch = "main"
        else:
            pattern = re.compile(r"^[a-zA-Z0-9._-]+$")
            if not pattern.match(branch):
                raise ValueError(
                    "Invalid branch name. Only alphanumeric characters, period, underscore and dash are allowed.")

        return model, branch

    def get_download_links_from_huggingface(self, model, branch, text_only=False, specific_file=None):
        page = f"/api/models/{model}/tree/{branch}"
        cursor = b""

        links = []
        sha256 = []
        classifications = []
        has_pytorch = False
        has_pt = False
        has_gguf = False
        has_safetensors = False
        is_lora = False
        link_file_size_list = []
        pytorch_indexs = []
        while True:

            url = f"{base}{page}" + (f"?cursor={cursor.decode()}" if cursor else "")
            print("url:",url)
            r = self.session.get(url, timeout=10)
            r.raise_for_status()
            content = r.content

            dict = json.loads(content)
            if len(dict) == 0:
                break
            for i in range(len(dict)):
                fname = dict[i]['path']
                if specific_file not in [None, ''] and fname != specific_file:
                    continue
                if not is_lora and fname.endswith(('adapter_config.json', 'adapter_model.bin')):
                    is_lora = True

                is_pytorch = re.match(r"(pytorch|adapter|gptq)_model.*\.bin", fname)
                is_safetensors = re.match(r".*\.safetensors", fname)
                is_pt = re.match(r".*\.pt", fname)
                is_gguf = re.match(r'.*\.gguf', fname)
                is_tiktoken = re.match(r".*\.tiktoken", fname)
                is_tokenizer = re.match(r"(tokenizer|ice|spiece).*\.model", fname) or is_tiktoken
                is_text = re.match(r".*\.(txt|json|py|md)", fname) or is_tokenizer
                if True or any((is_pytorch, is_safetensors, is_pt, is_gguf, is_tokenizer, is_text)):
                    if 'lfs' in dict[i]:
                        sha256.append([fname, dict[i]['lfs']['oid']])

                    if is_text:
                        links.append(f"https://huggingface.co/{model}/resolve/{branch}/{fname}")
                        classifications.append('text')
                        link_file_size_list.append(dict[i]["size"])
                        continue

                    if not text_only:
                        links.append(f"https://huggingface.co/{model}/resolve/{branch}/{fname}")
                        link_file_size_list.append(dict[i]["size"])
                        if is_safetensors:
                            has_safetensors = True
                            classifications.append('safetensors')
                        elif is_pytorch:
                            pytorch_indexs.append(len(links)-1)
                            has_pytorch = True
                            classifications.append('pytorch')
                        elif is_pt:
                            pytorch_indexs.append(len(links)-1)
                            has_pt = True
                            classifications.append('pt')
                        elif is_gguf:
                            has_gguf = True
                            classifications.append('gguf')

            cursor = base64.b64encode(f'{{"file_name":"{dict[-1]["path"]}"}}'.encode()) + b':50'
            cursor = base64.b64encode(cursor)
            cursor = cursor.replace(b'=', b'%3D')

        if (has_pytorch or has_pt) and has_safetensors:  
          links = np.array(links)
          link_file_size_list = np.array(link_file_size_list)
          links = np.delete(links, pytorch_indexs)
          link_file_size_list = np.delete(link_file_size_list, pytorch_indexs)
          links = links.tolist()
          link_file_size_list = link_file_size_list.tolist()
        is_llamacpp = has_gguf and specific_file is not None
        return links, sha256, is_lora, is_llamacpp,link_file_size_list
    def get_output_folder(self, model, branch, is_lora, is_llamacpp=False, base_folder=None):
        if base_folder is None:
            base_folder = 'models' if not is_lora else 'loras'

        # If the model is of type GGUF, save directly in the base_folder
        if is_llamacpp:
            return Path(base_folder)

        output_folder = f"{'_'.join(model.split('/')[-2:])}"
        if branch != 'main':
            output_folder += f'_{branch}'

        output_folder = Path(base_folder) / output_folder
        return output_folder

    def get_single_file(self, url, output_folder, start_from_scratch=False):
        filename = Path(url.rsplit('/', 1)[1])
        output_path = output_folder / filename
        headers = {}
        mode = 'wb'
        if output_path.exists() and not start_from_scratch:

            # Check if the file has already been downloaded completely
            r = self.session.get(url, stream=True, timeout=10)
            total_size = int(r.headers.get('content-length', 0))
            if output_path.stat().st_size >= total_size:
                return

            # Otherwise, resume the download from where it left off
            headers = {'Range': f'bytes={output_path.stat().st_size}-'}
            mode = 'ab'

        with self.session.get(url, stream=True, headers=headers, timeout=30) as r:
            r.raise_for_status()  # Do not continue the download if the request was unsuccessful
            total_size = int(r.headers.get('content-length', 0))
            block_size = 1024 * 1024  # 1MB
            with open(output_path, mode) as f:
                with tqdm.tqdm(total=total_size, unit='iB', unit_scale=True, bar_format='{l_bar}{bar}| {n_fmt:6}/{total_fmt:6} {rate_fmt:6}') as t:
                    count = 0
                    for data in r.iter_content(block_size):
                        t.update(len(data))
                        f.write(data)
                        if total_size != 0 and self.progress_bar is not None:
                            count += len(data)
                            self.progress_bar(float(count) / float(total_size), f"{filename}")

    def start_download_threads(self, file_list, output_folder, start_from_scratch=False, threads=4):
        thread_map(lambda url: self.get_single_file(url, output_folder, start_from_scratch=start_from_scratch), file_list, max_workers=threads, disable=True)

    def download_model_files(self, model, branch, links, sha256, output_folder, progress_bar=None, start_from_scratch=False, threads=1, specific_file=None, is_llamacpp=False):
        self.progress_bar = progress_bar

        # Create the folder and writing the metadata
        output_folder.mkdir(parents=True, exist_ok=True)

        if not is_llamacpp:
            metadata = f'url: https://huggingface.co/{model}\n' \
                       f'branch: {branch}\n' \
                       f'download date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'

            sha256_str = '\n'.join([f'    {item[1]} {item[0]}' for item in sha256])
            if sha256_str:
                metadata += f'sha256sum:\n{sha256_str}'

            metadata += '\n'
            (output_folder / 'huggingface-metadata.txt').write_text(metadata)

        if specific_file:
            print(f"Downloading {specific_file} to {output_folder}")
        else:
            print(f"Downloading the model to {output_folder}")
        # print("file_list:",links)
        self.start_download_threads(links, output_folder, start_from_scratch=start_from_scratch, threads=threads)

    def check_model_files(self, model, branch, links, sha256, output_folder):
        # Validate the checksums
        validated = True
        for i in range(len(sha256)):
            fpath = (output_folder / sha256[i][0])

            if not fpath.exists():
                print(f"The following file is missing: {fpath}")
                validated = False
                continue

            with open(output_folder / sha256[i][0], "rb") as f:
                bytes = f.read()
                file_hash = hashlib.sha256(bytes).hexdigest()
                if file_hash != sha256[i][1]:
                    print(f'Checksum failed: {sha256[i][0]}  {sha256[i][1]}')
                    validated = False
                else:
                    print(f'Checksum validated: {sha256[i][0]}  {sha256[i][1]}')

        if validated:
            print('[+] Validated checksums of all model files!')
        else:
            print('[-] Invalid checksums. Rerun download-model.py with the --clean flag.')
