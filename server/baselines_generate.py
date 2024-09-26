import os.path
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
sys.path.append('/ailab/user/wangwenhao/KnowledgeSG')
from dataset import *
from utils.utils import calculate_similarity, create_dataset_and_save
from utils.generation import *
from server.prompter import *
from server.inferencer import *


def icl_generate(args, inferencer, dataset_list, save_path, client_idx, round=100, use_vllm=False, icl=False,
                     self_instruct=False):
    dataset_shards_new = []
    for client_idx in range(client_idx[0], client_idx[1]):
        dataset_hf = dataset_list[client_idx]
        prompter = Prompter()
        generations, failed_generations = [], []
        prompt_list, dataset_sample_list, index_list_list = [], [], []
        # 生成prompt 以及获取fewshot的data sample
        for i in tqdm(range(round), desc='Generating prompts'):
            dataset_sample, index_list = prompter.gen_sample_list(dataset_hf)
            prompt = prompter.map_function(dataset_sample)

            prompt_list.append(prompt)  # for llama 2, default max model-len=4096
            dataset_sample_list.append(dataset_sample)
            index_list_list.append(index_list)

        outputs = generation(prompt_list, inferencer, use_vllm)

        # extract answer
        for i, prompt_output in enumerate(outputs):
            instruction, output = inferencer.extract_answer(prompt_output)
            # judge if instruction/output is None
            if not instruction or not output:
                continue

            # calc similarity
            sign = 0
            for sample in dataset_sample_list[i]:
                # for sample in dataset_sample:
                instruction_o = sample['instruction']
                try:
                    sim_ = calculate_similarity(instruction, instruction_o)
                    # filter by similarity # important
                    if sim_ > 0.52:
                        sign = 1
                except:
                    sign = 1
                    continue

            if (instruction and output is not None and sign == 0
                    and len(instruction) > 1 and len(output) > 1):
                generations.append({'instruction': instruction, 'output': output, 'index_list': index_list_list[i]})
            else:
                failed_generations.append({'instruction': instruction, 'output': output, 'index_list': index_list_list[i]})

        dataset_new = create_dataset_and_save(generations, client_idx, save_path)
        if args.save_failed_generation:
            dataset_failed = create_dataset_and_save(failed_generations, client_idx, save_path, sign=1)
        dataset_shards_new.append(dataset_new)

    return dataset_shards_new


def self_generate(args, inferencer, dataset_list, save_path, client_idx, round=100, use_vllm=False, icl=False,
                  self_instruct=False):
    dataset_shards_new = []
    for client_idx in range(client_idx[0], client_idx[1]):
        dataset_hf = dataset_list[client_idx]
        prompter = Prompter()

        generations, failed_generations = [], []
        prompt_list, dataset_sample_list, index_list_list = [], [], []

        # stage 1
        # 生成prompt 以及获取fewshot的data sample
        for i in tqdm(range(round), desc='Generating prompts'):
            dataset_sample, index_list = prompter.gen_sample_list(dataset_hf)
            prompt = prompter.map_self_instruct_1(dataset_sample)
            prompt_list.append(prompt)  # for llama 2, default max model-len=4096
            dataset_sample_list.append(dataset_sample)
            index_list_list.append(index_list)

        # 生成
        outputs = generation(prompt_list, inferencer, use_vllm)
        instruction_list = filter_instructions_by_similarity(inferencer, outputs, dataset_sample_list)

        # stage 2
        prompt_list, dataset_sample_list, index_list_list = [], [], []
        for instruction in tqdm(instruction_list, desc='Generating prompts'):
            dataset_sample, index_list = prompter.gen_sample_list(dataset_hf)

            prompt = prompter.map_self_instruct_2(dataset_sample, instruction, icl=icl)
            prompt_list.append(prompt[:4096])  # for llama 2, default max model-len=4096
            dataset_sample_list.append(dataset_sample)
            index_list_list.append(index_list)

        # 生成
        outputs = generation(prompt_list, inferencer, use_vllm)
        for i, (prompt_output, instruction) in enumerate(zip(outputs, instruction_list)):
            output = inferencer.extract_answer_output(prompt_output)
            if output:
                generations.append({'instruction': instruction, 'output': output, 'index_list': index_list_list[i]})

        dataset_new = create_dataset_and_save(generations, client_idx, save_path)
        if args.save_failed_generation:
            dataset_failed = create_dataset_and_save(failed_generations, client_idx, save_path, sign=1)
        dataset_shards_new.append(dataset_new)

    return dataset_shards_new


def recreate_dataset_only(args, inferencer, dataset_list, save_path, client_idx, round=100, use_vllm=False,
                          self_instruct=False):
    # for only using dataset['instruction'] and format to generate
    dataset_shards_new = []
    generations = []
    prompter = Prompter()
    # for dataset_idx, dataset_hf in enumerate(dataset_list):
    for client_idx in range(client_idx[0], client_idx[1]):
        dataset_hf = dataset_list[client_idx]
        instruction_list = dataset_hf['instruction']
        # stage 2
        prompt_list = []

        index_list_list = []
        if 'Fin' in args.dataset:
            choice = 'fin'
        else:
            choice = 'med'
        for instruction in tqdm(instruction_list, desc='Generating prompts'):
            if choice == 'fin':
                try:
                    instruction = instruction.split('}.')[1].strip()
                except:
                    instruction = ''
                prompt_list.append(prompter.map_fin(instruction[:512]))
            else:
                instruction = instruction[93:]
                if len(instruction) < 1:
                    print(':', instruction)

                prompt_list.append(prompter.map_med(instruction))
        # 生成
        outputs = generation(prompt_list, inferencer, use_vllm)

        for i, (prompt_output, instruction) in enumerate(zip(outputs, instruction_list)):
            output = prompt_output

            if output:
                generations.append({'instruction': instruction, 'output': output})
            else:
                print(instruction)

        dataset_new = create_dataset_and_save(generations, client_idx, save_path)
        dataset_shards_new.append(dataset_new)

    return dataset_shards_new


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="../llama2")
    parser.add_argument("--peft_model", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--round", type=int, default=100)
    parser.add_argument("--alg", type=str, default='')
    parser.add_argument("--dataset", type=str, default='')
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--use_vllm", action='store_true', default=False)
    parser.add_argument("--save_failed_generation", action='store_true', default=False)
    parser.add_argument("--choice", type=str, default='fin')
    args = parser.parse_args()

    assert not (args.alg == 'only' and '/llama2' in args.base_model and args.use_vllm)
    # modify to your root directory
    root_dir = './outputs/sampled_data'

    base_model = args.base_model
    peft_model = args.peft_model

    # prepare datasets
    data_path = f'{args.dataset}/train'
    print('data_path:', data_path)
    if not data_path.startswith('/'):
        data_path = os.path.join(root_dir, data_path)
    file_name_list = os.listdir(data_path)
    file_name_list.sort()
    print(file_name_list)
    dataset_list = [BaseDataset(os.path.join(data_path, x)).dataset for x in file_name_list]

    # set save_path
    if not args.save_path:
        args.save_path = f'./outputs/generated_data/{args.dataset}/{args.alg}/{args.id}'
    print('save_path: ', args.save_path)
    # set client_idx
    client_idx = (0, 1)
    if client_idx[1] - client_idx[0] > len(dataset_list):
        raise ValueError('too large client_idx')

    # merge model if needed
    if args.use_vllm and peft_model:
        target_path = f"./icl_full_model/"
        if os.path.exists(target_path):
            pass
        else:
            model_merge_save(base_model, peft_model, target_path)
        inferencer = Inferencer(target_path, use_vllm=True)
    else:
        tokenizer = None
        if 'dp' in args.alg:
            base_model_name = 'dp_model_path'

        inferencer = Inferencer(base_model, adapter_model_name=peft_model, tokenizer=tokenizer,
                                use_vllm=args.use_vllm)

    # === main ===
    if args.alg in ['icl', 'dp-gene']:
        icl_generate(args, inferencer, dataset_list, args.save_path, client_idx, args.round, use_vllm=args.use_vllm)
    elif args.alg in ['self-instruct-icl', 'dp-instruct-icl']:
        self_generate(args, inferencer, dataset_list, args.save_path, client_idx, args.round, icl=True,
                              use_vllm=args.use_vllm)
    elif args.alg in ['self-instruct', 'dp-instruct']:
        self_generate(args, inferencer, dataset_list, args.save_path, client_idx, args.round, icl=False,
                              use_vllm=args.use_vllm)
    elif args.alg in ['ksg', 'dp-ksg']:
        # data_list = self_generate(args, inferencer, dataset_list, args.save_path, client_idx, args.round, icl=True,
        #               use_vllm=args.use_vllm)
        #
    
        recreate_dataset_only(args, inferencer, dataset_list, args.save_path, client_idx, args.round,
                              use_vllm=args.use_vllm)