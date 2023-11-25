

from langchain.prompts import PromptTemplate


def get_model_type(model_path):
    if model_path:
        if model_path.lower().find("mistral") >= 0 and model_path.lower().find("instruct") >= 0:
            model_type = "mistral"
        elif model_path.lower().find("llama") >= 0 and model_path.lower().find("chat") >= 0:
            model_type = "llama2"
        elif model_path.lower().find("zephyr") >= 0:
            model_type = "zephyr"
        else:
            model_type = "other model"
    else:
        model_type = "other model"
    return model_type

def get_model_prompt_template(model_type="llama2"):
    if model_type == "other model":
        prompt_template = PromptTemplate.from_template(
            "{question}"
        )
    elif model_type == "llama2":
        prompt_template = PromptTemplate.from_template(
            "<s>[INST] {question} [/INST]"
        )
    elif model_type == "zephyr":
        prompt_template = PromptTemplate.from_template(
            "<|user|>\n{question}</s><|assistant|>\n"
        )
    elif model_type == "mistral":
        prompt_template = PromptTemplate.from_template(
            "<s>[INST] {question} [/INST]"
        )
    return prompt_template




def format_chat_history_prompt_for_llama2_7b_chat(chat_history):
    new_input = "<s>"
    for i, hist in enumerate(chat_history):
        if i % 2 == 0:
            new_input = new_input + "[INST] " + hist + " [/INST]"
        else:
            new_input = new_input + hist
    return new_input


def format_chat_history_prompt_for_mistral_7b_instruct(chat_history):
    new_input = "<s>"
    for i, hist in enumerate(chat_history):
        if i % 2 == 0:
            new_input = new_input + "[INST] " + hist + " [/INST]"
        else:
            new_input = new_input + hist
    return new_input


def format_chat_history_prompt_for_zephyr_7b_instruct(chat_history):
    new_input = ""
    for i, hist in enumerate(chat_history):
        if i % 2 == 0:
            new_input = new_input + "<|user|>\n" + hist + "</s>"
        else:
            if i==len(chat_history)-1:
                new_input = new_input + "<|assistant|>\n" + hist + ""
            else:
                new_input = new_input + "<|assistant|>\n" + hist + "</s>"
    return new_input

def get_chat_history_prompt(chat_history,model_type="llama2"):
    if model_type == "other model":
        prompt = ','.join(chat_history[:-2])
        prompt = prompt + chat_history[-2]
    elif model_type == "llama2":
        prompt = format_chat_history_prompt_for_llama2_7b_chat(chat_history)
    elif model_type == "zephyr":
        prompt = format_chat_history_prompt_for_zephyr_7b_instruct(chat_history)
    elif model_type == "mistral":
        prompt = format_chat_history_prompt_for_mistral_7b_instruct(chat_history)
    return prompt