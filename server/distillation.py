import os

from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizerFast, LlamaForCausalLM
import datasets
import vllm
import sys
import json
import random
from tqdm import tqdm
import argparse
from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
import torch
import openai

from .dataset import BaseDataset
from .utils import create_dataset_and_save


# Load Models
base_model = "/mnt/workspace/wenhaowang/hf_model/NousResearch/Llama-2-13b-hf"
peft_model = "/mnt/workspace/wenhaowang/hf_model/FinGPT/fingpt-sentiment_llama2-13b_lora"
tokenizer = LlamaTokenizerFast.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = LlamaForCausalLM.from_pretrained(base_model, trust_remote_code=True, device_map="auto", load_in_8bit=True, )
model = PeftModel.from_pretrained(model, peft_model)
model = model.eval()
device = 'cuda'

# modify this to your generated data of DP-finetuned model
data_path = '/GPFS/data/wenhaowang-1/KnowledgeSG/outputs/generated_data/Fin/llama2'

if not data_path.startswith('/'):
    data_path = os.path.join(root_dir, data_path)
file_name_list = os.listdir(data_path)
file_name_list.sort()
dataset_list = [BaseDataset(os.path.join(data_path, x)).dataset for x in file_name_list]


# data
dataset_hf = dataset_list[0]
instruction_list = dataset_hf['instruction']
# stage 2
prompt_list = []
generations = []
for instruction in tqdm(instruction_list, desc='Generating prompts'):
    try:
        instruction = instruction.split('}.')[1].strip()
    except:
        instruction = ''

    formated_input = "Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}\n" + instruction + "\nAnswer: "

    prompt_list.append(formated_input)


    inputs = tokenizer.encode(formated_input, return_tensors="pt", padding=True).to(device)
    # tokens = tokenizer(prompt, return_tensors='pt', padding=True, max_length=512)
    # outputs = model.generate(inputs=inputs, max_new_tokens=1024, do_sample=True, top_p=1.0, temperature=0.7)
    outputs = model.generate(inputs=inputs, max_new_tokens=20)
    a = tokenizer.decode(outputs[0])[len(formated_input) + 4:]
    b = a.split(tokenizer.unk_token)[0]

    generations.append({'instruction': instruction, 'response': b})
    # print("="*100)

dataset_new = create_dataset_and_save(generations, 0, '../outputs/generated_data/Fin/FinGPT/')
