from sentence_transformers import SentenceTransformer, util
import transformers
import datasets
import numpy as np
import mauve

from scipy.linalg import sqrtm
from scipy.spatial import distance
# from scipy.spatial import minkowski_distance
# from scipy.spatial.distance import euclidean


def get_dataset_embed(data_path, embed_model):
    dataset = datasets.load_dataset('json', data_files=data_path)['train']
    template = "#instruction:\n{}\n#response:\n{}"
    dataset = dataset.map(lambda example: {'ins+res': template.format(example['instruction'], example['response'])})
    sentences = dataset['ins+res']

    # Get embeddings of sentences
    embeddings = embed_model.encode(sentences)
    return embeddings


def calculate_mauve(embeddings1, embeddings2):
    # call mauve.compute_mauve using features obtained directly
    # p_feats and q_feats are `np.ndarray`s of shape (n, dim)
    out = mauve.compute_mauve(p_features=embeddings1, q_features=embeddings2)
    return out.mauve


def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="/data/LLMs/llama2")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--round", type=int, default=100)
    parser.add_argument("--alg", type=str, default='icl')
    parser.add_argument("--dataset", type=str, default='length')
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--use_vllm", action='store_true', default=False)
    args = parser.parse_args()

    # Download model
    embed_models = ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2", "all-mpnet-base-v2", "stsbroberta-base-v2",
                    "distilbert-base-nli-stsb-meantokens"]
    model = SentenceTransformer(embed_models[1])

    data_path_origin = f'/data/home/wangwenhao/Enron/{args.dataset}/train/client0_reform_name.json'

    data_path_target = f'/data/home/wangwenhao/Enron/generated_data/{args.dataset}/{args.alg}_save_500/client0.json'

    embeddings1 = get_dataset_embed(data_path_origin, model)
    embeddings2 = get_dataset_embed(data_path_target, model)
    print(len(embeddings1), len(embeddings2))
    print(calculate_fid(embeddings1[:100], embeddings2[:100]))

    print(calculate_mauve(embeddings1[:100], embeddings2[:100]))





