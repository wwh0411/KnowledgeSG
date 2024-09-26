import os
import json
from collections import defaultdict
import datasets
import random
import argparse


def merge_json_files(root_dir, output_dir):
    # 存储具有相同文件名的json文件路径
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_paths = defaultdict(list)

    # 遍历文件夹及子文件夹
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(subdir, file)
                file_paths[file].append(file_path)

    # 合并具有相同文件名的json文件
    for filename, paths in file_paths.items():
        merged_data = []
        for path in paths:
            try:
                with open(path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
            except:
                with open(path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                    data = []
                    for line in lines:
                        data.append(json.loads(line))
            if isinstance(data, list):
                merged_data.extend(data)
                print(len(merged_data))
            else:
                print(f"Warning: The file {path} does not contain a list of JSON objects.")

        # reformat data sample
        for sample in merged_data:
            if 'input' in sample.keys():
                if sample['input'] != '':
                    sample['instruction'] = sample['instruction'] + ' ' + sample['input']
                del sample['input']
                del sample['text']
            if 'output' in sample.keys():
                sample['response'] = sample['output']
                del sample['output']

        # 保存合并后的数据
        output_file = os.path.join(output_dir, filename)
        dataset_new = datasets.Dataset.from_list(merged_data)
        dataset_new.to_json(output_file)
        # with open(output_file, 'w', encoding='utf-8') as file:
        #     json.dump(merged_data, file, indent=4)


def random_select_and_save(src_dir, dest_dir, select_num):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # 遍历指定目录下的所有json文件
    for file in os.listdir(src_dir):
        if file.endswith('.json'):
            src_file_path = os.path.join(src_dir, file)
            dest_file_path = os.path.join(dest_dir, file)

            with open(src_file_path, 'r', encoding='utf-8') as src_file:
                lines = src_file.readlines()
                if len(lines) < select_num:
                    print(src_file_path)
                    select_num = len(lines)

                selected_lines = random.sample(lines, select_num)

            # for the ease of huggingface dataset load
            with open(dest_file_path, 'w', encoding='utf-8') as dest_file:
                for line in selected_lines:
                    dest_file.write(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="length")
    parser.add_argument("--alg", type=str, default="icl")
    parser.add_argument("--merge", action="store_true", default=True)
    parser.add_argument("--select", action="store_true", default=True)
    parser.add_argument("--select_num", type=int, default=500)
    args = parser.parse_args()
    data_dir = "./outputs/generated_data"
    origin_path = os.path.join(data_dir, args.dataset, args.alg)
    merge_temp_path = os.path.join(data_dir, args.dataset, args.alg + "_temp")
    select_save_path = os.path.join(data_dir, args.dataset, args.alg + "_save_" + str(args.select_num))
    
    if args.merge:
        merge_json_files(origin_path, merge_temp_path)
    if args.select:
        random_select_and_save(merge_temp_path, select_save_path, args.select_num)
    
