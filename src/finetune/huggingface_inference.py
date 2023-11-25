
from transformers import (AutoModelForCausalLM, AutoTokenizer,LlamaForCausalLM, LlamaTokenizer,pipeline,BitsAndBytesConfig)
from langchain.llms.llamacpp import LlamaCpp
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
import os
import torch
from langchain.prompts import PromptTemplate
from src.finetune.inference import Inference
class HuggingfaceInference(Inference):
    def __init__(self,model_path,max_new_tokens=256,temperature=0.7 ,top_p=0.95 ,top_k=1,repetition_penalty=1.15,using_4bit_quantization=True,low_cpu_mem_usage=False):
        self.model = None
        self.tokenizer = None
        self.hg_model = None
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.prompt_template = PromptTemplate.from_template(
            "{question}"
        )
        self.bnb_config = None
        if using_4bit_quantization:
            self.bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
        self.low_cpu_mem_usage = low_cpu_mem_usage
    def load_model(self):
        try:
          
            if self.model_path.split(os.sep)[-1].rfind("llama") >=0:
                self.tokenizer = LlamaTokenizer.from_pretrained(self.model_path)
                if self.bnb_config:
                    self.hg_model = LlamaForCausalLM.from_pretrained(self.model_path, device_map={"":0},quantization_config=self.bnb_config,torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,trust_remote_code=True)
                else:
                    self.hg_model = LlamaForCausalLM.from_pretrained(self.model_path, device_map={"": 0},torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,trust_remote_code=True)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                if self.bnb_config:
                    self.hg_model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map={"":0},quantization_config=self.bnb_config,torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,trust_remote_code=True)
                else:
                    self.hg_model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map={"": 0},torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,trust_remote_code=True)
            if not self.tokenizer.pad_token:
                if self.model_path.split(os.sep)[-1].lower().rfind("gpt2")>=0:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    self.hg_model.resize_token_embeddings(len(self.tokenizer))

        except Exception as e:
            return -1, e
        self.model = pipeline(
            "text-generation",
            model=self.hg_model,
            tokenizer=self.tokenizer,
            max_new_tokens = self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,top_k=self.top_k,do_sample=True,
            return_full_text=False,
            repetition_penalty=self.repetition_penalty,
            # return_dict_in_generate = True
        )
        return 0, ""
    def infer(self ,input):
        output = self.model(input)
        return  output[0]['generated_text'] if output else None
    def free_memory(self):
        if self.hg_model:
            del self.hg_model
            self.hg_model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        if self.model:
            del self.model
            self.model = None