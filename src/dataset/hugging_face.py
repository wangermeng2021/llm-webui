from datasets import Dataset, DatasetDict, load_dataset

from src.dataset.dataset import Dataset
import datasets
class HuggingFace(Dataset):
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        self.config={
            "dataset":{
                "train_dataset_path": "",
                "val_dataset_path": "",
                "huggingface_dataset": True,
                "prefxi1": "",
                "col1_name": "",
                "prefix2": "",
                "col2_name": "",
            },
            "model": {
                "base_model": "",
                "finetune_type": "",
                "huggingface_dataset": True,
                "lora_r": "",
                "lora_alpha": "",
                "lora_dropout": "",
                "lora_bias": "",

            },
            "training_arguments": {
                "epochs": "",
                "batch_size": "",
                "learning_rate": True,
                "optimizer": "",
                "gradient_checkpointing": True,
                "gradient_accumulation_steps": 1,
                "warmup_steps": "",
            }
        }
    def get_dataset(self,config):


        config["dataset"]["huggingface_dataset"]
        config["dataset"]["prefxi1"]
        config["dataset"]["col1_name"]
        config["dataset"]["prefix2"]
        config["dataset"]["col2_name"]



        huggingface_dataset = True
        prefxi1 = ""
        col1_name = ""
        prefix2 = ""
        col2_name = ""

        if config["dataset"]["huggingface_dataset"]:
            dataset_path = "samsum"
            dataset = datasets.load_dataset(dataset_path)
            train_split_name = ""
            val_split_name = ""
            for split in dataset.keys():
                if split.find("train"):
                    train_split_name = split
                elif split.find("val"):
                    val_split_name = split
            if train_split_name:
                train_dataset = dataset[train_split_name]
            if val_split_name:
                val_dataset = dataset[val_split_name]
        else:
            if config["dataset"]["train_dataset_path"].split(".")[-1] == "json":
                train_dataset = datasets.load_dataset("json", data_files=config["dataset"]["train_dataset_path"], split="train")
            elif config["dataset"]["train_dataset_path"].split(".")[-1] == "csv":
                train_dataset = datasets.load_dataset("parquet", data_files=config["dataset"]["train_dataset_path"], split="train")
            elif config["dataset"]["train_dataset_path"].split(".")[-1] == "parquet":
                train_dataset = datasets.load_dataset("parquet", data_files=config["dataset"]["train_dataset_path"], split="train")
            else:
                raise ValueError(f'Dataset format {config["dataset"]["train_dataset_path"].split(".")[-1]} is not yet supported.')

            if config["dataset"]["val_dataset_path"].split(".")[-1] == "json":
                val_dataset = datasets.load_dataset("json", data_files=config["dataset"]["val_dataset_path"], split="train")
            elif config["dataset"]["val_dataset_path"].split(".")[-1] == "csv":
                val_dataset = datasets.load_dataset("parquet", data_files=config["dataset"]["val_dataset_path"], split="train")
            elif config["dataset"]["val_dataset_path"].split(".")[-1] == "parquet":
                val_dataset = datasets.load_dataset("parquet", data_files=config["dataset"]["val_dataset_path"], split="train")
            else:
                raise ValueError(f'Dataset format {config["dataset"]["val_dataset_path"].split(".")[-1]} is not yet supported.')

        if "model_context_window" in self.config:
            context_window = self.config["model"]["model_context_window"]
        else:
            context_window = self.tokenizer.model_max_length


        train_dataset = train_dataset.map(lambda sample: self.tokenizer(
            self._generate_prompt(
                sample,
                self.tokenizer.eos_token),
            max_length=context_window,
            truncation=True,
        ))


        return dataset
