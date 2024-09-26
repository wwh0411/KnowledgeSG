import sys
import pandas as pd
import numpy as np
import random
import os
import json
import pdb
from datasets import load_dataset, Dataset


# Divide the entire dataset into a training set and a test set.
def partition_dataset(dataset_hf, num_clients):
    num_shards = 1
    # num_clients = 30
    partition_mode = 'length'
        
    if partition_mode == 'iid':
        # shuffle
        dataset_shuffle = dataset_hf.shuffle(seed=42)
        # get num_clients shards
        dataset_shard_list = [dataset_shuffle.shard(num_shards=num_clients, index=i)
            for i in range(num_clients)]

    elif partition_mode == 'length':
        # 对样本按长度排序
        # sorted_indices = np.argsort([len(example["instruction"]) for example in dataset_hf])
        sorted_dataset = sorted(dataset_hf, key=lambda x: len(x["instruction"]))
        
        # 将排序后的索引分成十个大致相等的部分
        num_samples = len(dataset_hf)
        chunk_size = num_samples // num_clients
        chunks = [sorted_dataset[i:i+chunk_size] for i in range(0, num_samples, chunk_size)]

        # 创建十个子数据集
        dataset_shard_list = [Dataset.from_list(chunk) for chunk in chunks]
        
    elif partition_mode == 'skew':
        # dataset_df = pd.Series(dataset_hf)
        column_cat = 'category' # column name maybe different
        num_category = len(set(dataset_hf[column_filter]))
        a = int(num_clients / num_category) # 30 / 8 = 3
        b = num_clients % num_category # 30 % 8 = 6
        # former6 huafen 3 + 1 = 4; latter2 huafen 3

        # turn dataset structure from huggingface into pandas 
        dataset_df = pd.DataFrame.from_dict(dataset_filter)

        # sort and group by column name 'category'
        dataset_df_sorted = dataset_df.sort_values(by=[column_cat]) 
        dataset_df_grouped = dataset_df_sorted.groupby(column_cat)

        # turn dataset of one group into shards
        dataset_shard_list = []
        for idx, (column_name, df) in enumerate(dataset_df_grouped):
            if idx < b:
                num_shards = a + 1
            else:
                num_shards = a
            shard_len = int(len(df) / num_shards)
            for i in range(num_shards):
                df_shard = df[i * shard_len: (i + 1) * shard_len]
                # turn dataset structure back to hf
                dataset_shard = Dataset.from_pandas(df_shard).remove_columns('__index_level_0__')
                dataset_shard_list.append(dataset_shard)

        # shards = np.array_split(pd_dataset.index.values, int(num_shards * num_clients))
        print(len(dataset_shard_list))
        # num_shard_per_client = int(len(dataset_shard_list) / num_clients)
        # shuffle differnet shards

    return dataset_shard_list
        
