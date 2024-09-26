import sys
import copy
import os
from tqdm import tqdm
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training, PeftModel

from utils import *
from fed import *
from gc_generation import recreate_dataset
from config import get_config, save_config, get_model_config, get_training_args

from datasets import load_dataset
import dp_transformers
import subprocess
import re
import torch

def get_gpu_memory_usage():
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        gpu_memory_usage = [int(x) for x in output.decode('utf-8').strip().split('\n')]
        return gpu_memory_usage
    except Exception as e:
        print("Error occurred while getting GPU memory usage:", e)
        return None


# ===== Define the arguments =====
script_args, fed_args, privacy_args, peft_config, auth_yaml = get_config()
training_args = get_training_args(script_args, script_args.learning_rate)
save_config(script_args, fed_args, privacy_args)
print(script_args, fed_args, privacy_args)
verbose = script_args.v

# ===== Load the dataset =====
if fed_args.split_strategy == 'cache':
    local_datasets = []
    local_dir = script_args.local_data_dir
    for i in range(fed_args.num_clients):
        local_file_path = local_dir + f'client{i}.json'
        data_shard = load_dataset('json', data_files=local_file_path, split='train')#.rename_column("output", "response")
        dataset = data_shard.shuffle(seed=2023)
        num_sample = min(len(dataset), script_args.dataset_sample)
        dataset = dataset.select(range(num_sample))
        local_datasets.append(dataset)
else:
    dataset = get_dataset(script_args.dataset_name, script_args.local_data_dir)
    dataset = process_sft_dataset(script_args.dataset_name, dataset, script_args.dataset_sample)

    # ===== Split the dataset into clients =====
    local_datasets = split_dataset(fed_args, script_args, dataset)
print(len(local_datasets))
sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]
print(sample_num_list)
print(local_datasets[0][0])

# ===== Get model config =====
device_map, quantization_config, torch_dtype = get_model_config(script_args)

# model load
class model_loader:
    def __init__(self):
        
        model = AutoModelForCausalLM.from_pretrained(
            script_args.model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=script_args.trust_remote_code,
            torch_dtype=torch_dtype,
            load_in_8bit=script_args.load_in_8bit,
            token=auth_yaml["auth_token"] if auth_yaml else None
        )

        if script_args.load_in_8bit or script_args.load_in_4bit:
            model = prepare_model_for_kbit_training(
                    model, use_gradient_checkpointing=training_args.gradient_checkpointing if not script_args.use_dp else False
                )
        peft_model = script_args.peft_model
        if peft_model:
            print("using:", peft_model)
            model = PeftModel.from_pretrained(model, peft_model)
        else:
            model = get_peft_model(model, peft_config)
        
        

        model.print_trainable_parameters()
        self.model = model
        

# print(f">> Init: fixed param: {model.state_dict()['base_model.model.model.layers.0.self_attn.q_proj.weight'][0]}, learned: {model.state_dict()['base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight'][0]}")
model_load = model_loader()
# model_load.__init__()
#print(model_load.attribute)
model = model_load.model
# ===== Define the global and local models =====
global_dict = copy.deepcopy(get_peft_model_state_dict(model))
local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]
proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)

# del model
print('before all:', global_dict['base_model.model.model.layers.1.self_attn.v_proj.lora_B.weight'][20])
if script_args.use_dp:
    del model_load.model
    torch.cuda.empty_cache()

# calculate A
if fed_args.fed_alg == 'a':
    from feature.test_similarity import get_A
    origin_data_path = '/mnt/workspace/wenhaowang/Enron/generated_data/chatbot/chatbot_merge/chatbot_origin_500/'
    origin_datasets = [load_dataset('json', data_files=origin_data_path + f'client{i}.json', split='train') 
        for i in range(fed_args.num_clients)]
            
    proxy_dict = [get_A(origin_datasets[i], local_datasets[i]) for i in range(fed_args.num_clients)]
    print(proxy_dict)
else:
    pass
# ===== Define the tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(
    script_args.model_name, 
    use_fast=False, 
    padding_side="right", 
    token=auth_yaml["auth_token"] if auth_yaml else None
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token   # following vicuna

# ===== Define the formatting function =====
formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]  # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]`
if fed_args.fed_alg == 'dp' or (fed_args.fed_alg).startswith('dplocal'):
    data_collator = dp_transformers.dp_utils.DataCollatorForPrivateCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
else:
    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

# ===== Start federated training =====
training_loss = [[] for i in range(fed_args.num_clients)]

for round in tqdm(range(fed_args.num_rounds)):

    clients_this_round = get_clients_this_round(fed_args, round)

    print(f">> ==================== Round {round+1} : {clients_this_round} ====================")
    
    # if round !=0 and round % 60 == 0:
    #     # generating synthetic
    #     for client in range(fed_args.num_clients):
    #         local_datasets[client] = recreate_dataset(model, tokenizer, local_datasets[client])
            

    # normal federated train
    for client in range(fed_args.num_clients):
        
        if client not in clients_this_round:
            training_loss[client].append(-1)        # -1 is an indicator of not training
            continue
        # model_load_ = model_loader()
        # model = model_load_.model
        if verbose:
            print('before train:', global_dict['base_model.model.model.layers.1.self_attn.v_proj.lora_B.weight'][20])
     

        set_peft_model_state_dict(model, global_dict)   # sync the global model to the local model

        sub_dataset = get_dataset_this_round(local_datasets[client], round, fed_args, script_args)  # get the required sub-dataset for this round
        new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 1e-6) # manually schedule the learning rate
        training_args = get_training_args(script_args, new_lr)

        trainer = get_fed_local_sft_trainer(
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            # local_dataset=sub_dataset,
            local_dataset=local_datasets[client] if script_args.use_dp else sub_dataset,
            formatting_prompts_func=formatting_prompts_func,
            data_collator=data_collator,
            global_dict=global_dict,
            fed_args=fed_args,
            script_args=script_args,
            privacy_args=privacy_args,
            local_auxiliary=auxiliary_model_list[client],
            global_auxiliary=global_auxiliary,
            round=round+1, # add
        )

        if fed_args.fed_alg == 'scaffold':
            trainer.add_callback(SCAFFOLD_Callback(trainer.correction, model))

        # train
        results = trainer.train()
        
        training_loss[client].append(results.training_loss)
        
        if fed_args.fed_alg == 'scaffold':
            auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()

        local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model))   # copy is needed!
        print('after train_update:', local_dict_list[client]['base_model.model.model.layers.1.self_attn.v_proj.lora_B.weight'][20])
        if script_args.use_dp:
            del model_load_
        gpu_memory_usage = get_gpu_memory_usage()
        if verbose:
            if gpu_memory_usage is not None:
                for i, usage in enumerate(gpu_memory_usage):
                    print(f"GPU {i}: {usage} MiB")
    # ===== Aggregate the local models =====
    global_dict, global_auxiliary = global_aggregate(fed_args, global_dict, local_dict_list, sample_num_list, clients_this_round, round, proxy_dict=proxy_dict, opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict))
    if verbose:
        print('after aggregate:', global_dict['base_model.model.model.layers.1.self_attn.v_proj.lora_B.weight'][20])
        
    
    # for key in global_dict.keys():
    #     global_dict[key] = sum([local_dict_list[client][key] for client in clients_this_round]) / len(clients_this_round)

    set_peft_model_state_dict(model, global_dict)   # update global model
    
    # ===== Save the model =====
    if (round+1) % 10 == 0:
        # trainer.save_model(os.path.join(script_args.output_dir, f"checkpoint-{round+1}"))
        model.save_pretrained(os.path.join(script_args.output_dir, f"checkpoint-{round+1}"))
        # base_model = AutoModelForCausalLM.from_pretrained(
        #     script_args.model_name,
        #     quantization_config=quantization_config,
        #     device_map=device_map,
        #     trust_remote_code=script_args.trust_remote_code,
        #     torch_dtype=torch_dtype,
        #     load_in_8bit=script_args.load_in_8bit,
        #     token=auth_yaml["auth_token"] if auth_yaml else None
        # )
        # peft_model_temp = PeftModel.from_pretrained(base_model, os.path.join(script_args.output_dir, f"checkpoint-{round+1}"))
        # a = get_peft_model_state_dict(peft_model_temp)
        # print('after save:', a['base_model.model.model.layers.1.self_attn.v_proj.lora_B.weight'][20])
    
    # del model_load
    if verbose:
        gpu_memory_usage = get_gpu_memory_usage()
        if gpu_memory_usage is not None:
            for i, usage in enumerate(gpu_memory_usage):
                print(f"GPU {i}: {usage} MiB")

    np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))