import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import argparse


def create_dataset_and_save(generations, i, save_path, sign=0):
    dataset_new = datasets.Dataset.from_list(generations)
    print(dataset_new)
    print('####finish generating####')
    if sign == 0:
        file_name = save_path + f'/client{i}.json'
    else:
        file_name = save_path + '_failed/' + f'/client{i}.json'
    dataset_new.to_json(file_name)
    return dataset_new


def calculate_similarity(text1, text2):
    # 使用TF-IDF转换文本
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    # 计算余弦相似度
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return cosine_sim[0][0]


def map_clean_response(example):
    response = example['response']
    response = response.split('##')[0]
    print(response)
    example['response'] = response
    return example


def model_merge_save(base_model_name, lora_path, target_model_path):
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    peft_model = PeftModel.from_pretrained(base_model, lora_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)

    model = peft_model.merge_and_unload()

    model.save_pretrained(target_model_path)
    tokenizer.save_pretrained(target_model_path)