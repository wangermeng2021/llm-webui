import torch
import transformers,os
from peft import (LoraConfig, PeftModel, get_peft_model,
                  prepare_model_for_kbit_training)
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, LlamaForCausalLM, LlamaTokenizer)
# from src.dataset.raw_text_dataset import RawTextDataset
import datasets
from src.finetune.peft_trainer import PeftTrainer
from src.utils.lora_target_modules import LORA_TARGET_MODULES
from datetime import datetime





class TrainingStatus():
    def __init__(self):
        self.status = -1
TRAINING_STATUS = TrainingStatus()

class QloraTrainer(PeftTrainer):

    def __init__(self, config: dict):
        self.config = config
        self.tokenizer = None
        self.base_model = None
        self.merged_model = None
        self.dataset = None
        self.fused_model = None
        self.train_dataset = None
        self.val_dataset = None
        self.logging_callback = self.LoggingCallbacks()
        print("config:",config)
    def load_dataset(self):
        if self.config["dataset"]["hg_dataset_dir"]:
            if os.path.exists(os.path.join(self.config["dataset"]["hg_dataset_dir"],"dataset_infos.json")):
                if self.config["dataset"]["hg_train_dataset"]:
                    self.train_dataset= datasets.load_dataset(self.config["dataset"]["hg_dataset_dir"],split=self.config["dataset"]["hg_train_dataset"])
                if self.config["dataset"]["hg_val_dataset"]:
                    self.val_dataset = datasets.load_dataset(self.config["dataset"]["hg_dataset_dir"],split=self.config["dataset"]["hg_val_dataset"])
            elif os.path.exists(os.path.join(self.config["dataset"]["hg_dataset_dir"],"dataset_dict.json")):
                if self.config["dataset"]["hg_train_dataset"]:
                    self.train_dataset = datasets.load_from_disk(
                        self.config["dataset"]["hg_dataset_dir"] + "/" + self.config["dataset"]["hg_train_dataset"])
                if self.config["dataset"]["hg_val_dataset"]:
                    self.val_dataset = datasets.load_from_disk(
                        self.config["dataset"]["hg_dataset_dir"] + "/" + self.config["dataset"]["hg_val_dataset"])
            else:
                raise ValueError(
                    f'Invalid Dataset format {self.config["dataset"]["hg_dataset_dir"]}.')
        else:

            if self.config["dataset"]["local_dataset_dir"]:
                if os.path.exists(os.path.join(self.config["dataset"]["local_dataset_dir"], "dataset_infos.json")):
                    if self.config["dataset"]["local_train_set"]:
                        self.train_dataset = datasets.load_dataset(self.config["dataset"]["local_dataset_dir"],
                                                                   split=self.config["dataset"]["local_train_set"])
                    if self.config["dataset"]["local_val_set"]:
                        self.val_dataset = datasets.load_dataset(self.config["dataset"]["local_dataset_dir"],
                                                                   split=self.config["dataset"]["local_val_set"])
                elif os.path.exists(os.path.join(self.config["dataset"]["local_dataset_dir"], "dataset_dict.json")):
                    if self.config["dataset"]["local_train_set"]:
                        self.train_dataset = datasets.load_from_disk(
                            self.config["dataset"]["local_dataset_dir"] + "/" + self.config["dataset"]["local_train_set"])
                    if self.config["dataset"]["local_val_set"]:
                        self.val_dataset = datasets.load_from_disk(
                            self.config["dataset"]["local_dataset_dir"] + "/" + self.config["dataset"]["local_val_set"])
                else:
                    raise ValueError(
                        f'Invalid Dataset format {self.config["dataset"]["local_dataset_dir"]}.')


        if self.config["dataset"]["max_length"] == "Model Max Length":

            if self.config["model"]["base_model_name"].rfind("llama") >= 0:
                context_window = 1024*4
            elif self.config["model"]["base_model_name"].rfind("mistral") >= 0:
                context_window = 1024*4
            elif self.config["model"]["base_model_name"].rfind("zephyr") >= 0:
                context_window = 1024*4
            else:
                context_window = self.tokenizer.model_max_length
                if self.tokenizer.model_max_length == int(1e30):
                    context_window = 1024
        else:
            context_window = self.config["dataset"]["max_length"]
        print("context_window:",context_window)
        self.train_dataset = self.train_dataset.map(lambda sample: self.tokenizer(
            self.generate_prompt(
                sample,
                self.tokenizer.eos_token),
            max_length=context_window,
            truncation=True,
            # padding=True
        ))
        if self.val_dataset:
            self.val_dataset = self.val_dataset.map(lambda sample: self.tokenizer(
                self.generate_prompt(
                    sample,
                    self.tokenizer.eos_token),
                max_length=context_window,
                truncation=True,
                padding=True
            ))
    def generate_prompt(self,sample,eos_token):

        prompt = self.config["dataset"]["prefix1"]+sample[self.config["dataset"]["datatset_col1"]]+\
                 self.config["dataset"]["prefix2"] + sample[self.config["dataset"]["datatset_col2"]]+eos_token
        # print("prompt:",prompt)
        return prompt

    def load_model(self):

        if self.config["model"]["fine_tuning_type"] == "QLoRA":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        elif self.config["model"]["fine_tuning_type"] == "LoRA":
            bnb_config = None
        try:
            if self.config["model"]["base_model_name"].rfind("llama")>=0:
                self.tokenizer = LlamaTokenizer.from_pretrained(self.config["model"]["base_model_path"])
                self.base_model = LlamaForCausalLM.from_pretrained(self.config["model"]["base_model_path"], quantization_config=bnb_config, device_map={"":0},trust_remote_code=True)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.config["model"]["base_model_path"])
                self.base_model = AutoModelForCausalLM.from_pretrained(self.config["model"]["base_model_path"], quantization_config=bnb_config, device_map={"":0},trust_remote_code=True)
        except Exception as e:
            return -1,e
        if not self.tokenizer.pad_token:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.base_model.resize_token_embeddings(len(self.tokenizer))
        if self.config["training"]["gradient_checkpointing"] and not self.config["model"]["base_model_name"].rfind("phi")>=0:
            # self.base_model.gradient_checkpointing_enable()
            self.base_model = prepare_model_for_kbit_training(self.base_model,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
        else:
            self.base_model = prepare_model_for_kbit_training(self.base_model, use_gradient_checkpointing=False,gradient_checkpointing_kwargs={'use_reentrant':False})
        if  self.config["model"]["base_model_name"].lower().rfind("llama")>=0 or \
            self.config["model"]["base_model_name"].lower().rfind("mistral") >= 0 or \
            self.config["model"]["base_model_name"].lower().rfind("zephyr") >= 0:
            target_modules = LORA_TARGET_MODULES["llama"]
            task_type = "CAUSAL_LM"
        elif self.config["model"]["base_model_name"].lower().find("falcon") >= 0:
            target_modules = LORA_TARGET_MODULES["falcon"]
            task_type = "CAUSAL_LM"
        elif self.config["model"]["base_model_name"].lower().find("gpt2") >= 0:
            target_modules = LORA_TARGET_MODULES["gpt2"]
            task_type = "CAUSAL_LM"
        elif self.config["model"]["base_model_name"].lower().find("phi") >= 0:
            target_modules = ["Wqkv", "out_proj"]
            task_type = "CAUSAL_LM"
        else:
            raise ValueError(f'{self.config["model"]["base_model_name"]} is not yet supported.')
            #T5,bart, task_type = "SEQ_2_SEQ_LM" ,AutoModelForSeq2SeqLM
        
        lora_config = LoraConfig(
            r=self.config["model"]["lora_r"],
            lora_alpha=self.config["model"]["lora_alpha"],
            target_modules=target_modules,
            lora_dropout=self.config["model"]["lora_dropout"],
            bias=self.config["model"]["lora_bias"],
            task_type=task_type,
        )
        self.fused_model = get_peft_model(self.base_model, lora_config)
        # self.fused_model.gradient_checkpointing = True
        return 0,""
    def train(self):
        self.run_name = datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
        logging_dir = os.path.join(self.config["training"]["root_dir"],"runs", self.run_name,"tensorboard")
        run_output_model_name = self.config['model']['base_model_name'].replace('/', '_')
        output_model_dir = os.path.join(self.config["training"]["root_dir"],"runs",  self.run_name,"output_model", run_output_model_name + "_adapter")
        checkpoint_dir = os.path.join(self.config["training"]["root_dir"],"runs",  self.run_name)
        self.trainer = transformers.Trainer(
            model=self.fused_model,
            train_dataset=self.train_dataset,
            eval_dataset= self.val_dataset if self.val_dataset else None,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=self.config["training"]["batch_size"],
                gradient_accumulation_steps=self.config["training"]["gradient_accumulation_steps"],
                warmup_steps=self.config["training"]["warmup_steps"],
                num_train_epochs=self.config["training"]["epochs"],
                learning_rate=self.config["training"]["learning_rate"],
                fp16=True,
                output_dir=checkpoint_dir,
                report_to="tensorboard",
                optim=self.config["training"]["optimizer"],
                lr_scheduler_type=self.config["training"]["lr_scheduler_type"],
                load_best_model_at_end=True if self.val_dataset else False,
                save_strategy="steps",
                save_steps = self.config["training"]["eval_steps"],
                save_total_limit=1,
                evaluation_strategy="steps" if self.val_dataset else "no",
                eval_steps=self.config["training"]["eval_steps"],  # eval interval
                per_device_eval_batch_size=1,
                # eval_steps=10,  # eval interval
                logging_steps=100,#self.config["training"]["eval_steps"]
                # run_name=self.run_name,
                logging_dir=logging_dir,
            ),

            callbacks=[self.logging_callback,transformers.EarlyStoppingCallback(early_stopping_patience=self.config["training"]["early_stopping_patience"]) ] if self.config["training"]["early_stopping_patience"]>0 else [self.logging_callback],
            data_collator=transformers.DataCollatorForLanguageModeling(self.tokenizer, mlm=False),

        )

        self.fused_model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        try:
            self.trainer.train()
        except Exception as e:
            return -1,e
        # model_save_path = f"{self.config['training']['output_dir']}/{self.config['model']['base_model_name']}_adapter"
        self.trainer.save_model(output_model_dir)
        return 0,""
    def merge_and_save(self):

        if self.config["model"]["base_model_name"].rfind("llama")>=0:
            base_model = LlamaForCausalLM.from_pretrained(self.config["model"]["base_model_path"], device_map="cpu",trust_remote_code=True)
        else:
            base_model = AutoModelForCausalLM.from_pretrained(self.config["model"]["base_model_path"], device_map="cpu",trust_remote_code=True)
        run_output_model_name = self.config['model']['base_model_name'].replace('/', '_')
        output_adapter_model_dir = os.path.join(self.config["training"]["root_dir"], "runs", self.run_name, "output_model",
                                        run_output_model_name + "_adapter")

        model = PeftModel.from_pretrained(base_model, output_adapter_model_dir)

        merged_model = model.merge_and_unload()
        run_output_model_name = self.config['model']['base_model_name'].replace('/', '_')
        output_merged_model_dir = os.path.join(self.config["training"]["root_dir"], "runs", self.run_name, "output_model","merged_"+run_output_model_name,"ori")
        merged_model.save_pretrained(output_merged_model_dir)
        self.tokenizer.save_pretrained(output_merged_model_dir)

    def _print_trainable_parameters(self, model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )


    class LoggingCallbacks(transformers.TrainerCallback):
        # current_step = 0
        # max_steps = 0

        def on_step_begin(self, args: transformers.TrainingArguments, state: transformers.TrainerState,
                          control: transformers.TrainerControl, **kwargs):
            pass

        def on_step_end(self, args: transformers.TrainingArguments, state: transformers.TrainerState,
                        control: transformers.TrainerControl, **kwargs):
            global TRAINING_STATUS
            if TRAINING_STATUS.status == 1:
                control.should_epoch_stop = True
                control.should_training_stop = True
            else:
                self.max_steps = state.max_steps
                self.current_step = state.global_step

        def on_log(self, args: transformers.TrainingArguments, state: transformers.TrainerState,
                   control: transformers.TrainerControl, logs, **kwargs):
            pass

    def free_memroy(self):
        try:
            del self.fused_model
            del self.tokenizer
            del self.base_model
            del self.trainer
            torch.cuda.empty_cache()
        except Exception as e:
            print("Free memory error:",e)
