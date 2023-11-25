
from transformers import (AutoModelForCausalLM, AutoTokenizer,LlamaForCausalLM, LlamaTokenizer,pipeline)
import os
class InferenceModel():
    def __init__(self,model_path,prefix1="",prefix2="",max_length=1024 ,temperature=0.7 ,top_p=0.95 ,repetition_penalty=1.15 ,top_k=1,
                 do_sample=True,num_beams=1,penalty_alpha=0.6

                 ):

        self.pipe = None
        self.tokenizer = None
        self.model = None
        self.model_path = model_path

        self.prefix1 = prefix1
        self.prefix2 = prefix2
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty

    def load_model(self):
        try:
            if self.model_path.split(os.sep)[-1].rfind("llama") >=0:
                self.tokenizer = LlamaTokenizer.from_pretrained(self.model_path)
                self.model = LlamaForCausalLM.from_pretrained(self.model_path, device_map={"":0},trust_remote_code=True)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map={"":0},trust_remote_code=True)
            if not self.tokenizer.pad_token:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.model.resize_token_embeddings(len(self.tokenizer))
        except Exception as e:
            return -1, e
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty
        )
        return 0, ""

    def generate_prompt(self, input1):
        return self.prefix1+input1+self.prefix2
    def infer(self ,input1 ):
        prompt = self.generate_prompt(input1)
        print(prompt)
        gen_output = self.pipe(prompt)
        # print("output:",gen_output[0]['generated_text'] if gen_output else "The generation is finished!")
        return  gen_output[0]['generated_text'] if gen_output else "The generation is finished!"

    def free_memory(self):
        del self.model
        del self.tokenizer
        del self.pipe