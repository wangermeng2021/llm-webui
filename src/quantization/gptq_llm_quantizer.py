try:
  from transformers import AutoModelForCausalLM, AutoTokenizer
  from optimum.gptq import GPTQQuantizer, load_quantized_model
  import torch
  import datasets
  class GptqLlmQuantizer():
      def __init__(self,model_path,dataset_path,prefix1,prefix2,datatset_col1,datatset_col2):
          self.model_path = model_path
          self.dataset_path = dataset_path
          self.prefix1 = prefix1;
          self.prefix2 = prefix2;
          self.datatset_col1 = datatset_col1;
          self.datatset_col2 = datatset_col2;

      def load_model(self):
          try:
              self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
              self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.float16, device_map={"":0},trust_remote_code=True)
              self.eos_token = self.tokenizer.eos_token
          except Exception as e:
              return -1,e
          return 0,""
      def quantize(self,dataset,saved_model_dir):
          quantizer = GPTQQuantizer(bits=4, dataset=dataset)
          quantized_model = quantizer.quantize_model(self.model, self.tokenizer)
          quantizer.save(quantized_model, saved_model_dir)

      def prepare_dataset(self):

          if self.dataset_path.split(".")[-1] == "json":
              self.train_dataset = datasets.load_dataset("json", data_files=self.dataset_path, split="all")
          elif self.dataset_path.split(".")[-1] == "csv":
              self.train_dataset = datasets.load_dataset("parquet", data_files=self.dataset_path, split="all")
          elif self.dataset_path.split(".")[-1] == "parquet":
              self.train_dataset = datasets.load_dataset("parquet", data_files=self.dataset_path, split="all")
          else:
              raise ValueError(f'Dataset format {self.dataset_path.split(".")[-1]} is not yet supported.')

          output_text_list = []
          for sample in self.train_dataset:
              output_text = self.prefix1 + sample[self.datatset_col1] + self.prefix2 + sample[self.datatset_col2]+self.eos_token
              output_text_list.append(output_text)
          return output_text_list
except:
  pass