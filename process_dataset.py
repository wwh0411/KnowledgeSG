from dataset import *
import random
import argparse


def map_medical_chatbot(example):
    example['instruction'] = example['Patient']
    example['response'] = example['Doctor']
    return example


def map_alpaca(example):
    example['instruction'] = example['instruction'] + ' ' + example['input']
    example['response'] = example['output']
    return example


def map_chatbot(example):
    example['instruction'] = example["conversation_a"][0]["content"]
    example['response'] = example["conversation_a"][1]["content"]
    return example


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="length")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--select_num", type=int, default=500)
    args = parser.parse_args()

    path = args.dataset_path

    # load processor from dataset directory
    DataProcessor = BaseDataset(path)
    if args.dataset == 'chatbotarena':
        DataProcessor.dataset = DataProcessor.dataset.filter(lambda example: example['language'] == 'English')
    dataset_ = DataProcessor.random_select(num_choice=args.select_num)

    # map and remove columns
    if args.dataset == 'medchatbot':
        dataset_ = dataset_.map(map_medical_chatbot, remove_columns=['Description', 'Patient', 'Doctor'])
    elif args.dataset == 'codealpaca':
        dataset_ = dataset_.map(map_alpaca, remove_columns=['input', 'output'])
    elif args.dataset == 'chatbotarena':
        dataset_ = dataset_.map(map_chatbot,
                                remove_columns=['question_id', 'model_a', 'model_b', 'winner', 'judge',
                                                'turn', 'anony', 'language', 'tstamp', 'toxic_chat_tag',
                                                'conversation_a', 'conversation_b', 'openai_moderation'])

    dataset_.to_json(f'./outputs/sampled_data/{args.dataset}_{args.select_num}/train/client0.json')
