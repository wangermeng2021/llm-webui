
from transformers import (AutoModelForCausalLM, AutoTokenizer,LlamaForCausalLM, LlamaTokenizer,pipeline)
from langchain.llms.llamacpp import LlamaCpp
import os
import torch
from langchain.prompts import PromptTemplate
from src.finetune.inference import Inference
class LlamaCppInference(Inference):
    def __init__(self,model_path,max_new_tokens=256,temperature=0.7 ,top_p=0.95 ,top_k=1,repetition_penalty=1.15,n_gpu_layers=35, n_ctx=4048,verbose=False):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.prefix1 = ""
        self.prefix2 = ""
        self.model = None

    def load_model(self):
        load_model_status = 0
        msg = None
        try:
            self.model = LlamaCpp(model_path=self.model_path, n_gpu_layers=35, n_ctx=4096,max_tokens=self.max_new_tokens, temperature=self.temperature,
                                   verbose=False, top_k=self.top_k, top_p=self.top_p,repeat_penalty=self.repetition_penalty)
        except Exception as e:
            load_model_status = -1
            msg = e
        return load_model_status, msg
    def infer(self ,input):
        return self.model(input)


    def free_memory(self):
        if self.model:
            del self.model
            self.model = None