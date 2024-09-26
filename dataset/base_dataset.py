from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType
import torch
from datasets import load_dataset
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import random
import json

from .partition import partition_dataset


class BaseDataset:
    def __init__(self, data_path):
        # get dataset name
        # get dataset path
        # configure to your own raw data path
        self.path = data_path

        # load from local file
        try:
            self.dataset = load_dataset(self.path)['train']
        except:
            file_format = self.path.split('.')[-1]
            if file_format == 'jsonl':
                file_format = 'json'
            self.dataset = load_dataset(file_format, data_files=self.path)['train']

        # set sample size per client to max_samples

    def prepare(self, num_clients, dataset=None, map_function=None, remove_columns=None):
        if not dataset:
            dataset = self.dataset
            
        # get train val split  
        if map_function:
            dataset = dataset.map(map_function)
        if remove_columns:
            dataset = dataset.remove_columns(remove_columns)
        # train_ds_list.append(data_shard)
        # partition dataset into several shards
        train_ds_list = partition_dataset(dataset, num_clients)
        return train_ds_list

    def set_size(self, dataset=None, max_size=1000):
        # reduce the dataset size if too large
        if not dataset:
            dataset = self.dataset
        dataset = dataset.filter(
            lambda example, idx: idx < max_size, with_indices=True)
        return dataset

    def random_select(self, dataset=None, num_choice=1):
        if not dataset:
            dataset = self.dataset
        num_length = len(dataset)
        if num_choice > num_length:
            raise ValueError('num_choice outranges num_length')
        dataset = dataset.select(random.sample([i for i in range(num_length)], num_choice))
        return dataset

    def save_data_shards(self, data_shards, data_dir=None):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        for i, shard in enumerate(data_shards):
            file_name = data_dir + f'/client{i}.json'
            shard.to_json(file_name)
