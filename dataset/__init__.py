from .base_dataset import BaseDataset
from .partition import partition_dataset


def get_dataloader(args, configs, prompter, tokenizer):
    if configs['dataset']['train_name'] != 'domain':
        dataset = BaseDataset(args, configs)
    else:
        dataset = DomainDataset(args, configs)
    # mean, max = dataset.count(tokenizer, 'instruction')

    # federated dataloader
    train_dls, val_dls = dataset.prepare(prompter.generate_and_tokenize_prompt)
    client_num_samples = [len(x) for x in train_dls]

    return train_dls, val_dls, client_num_samples