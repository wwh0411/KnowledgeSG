This is the official repository for the EMNLP 2024 paper "KnowledgeSG: Privacy-Preserving Synthetic Text Generation With Knowledge Distillation From Server"

## Basic Usage
Process and sample a huggingface dataset
```
python process_dataset.py --dataset <Dataset Name> --dataset_path <Your Dataset Path>
```
DP-Finetune of the base model.  
Refer to https://github.com/microsoft/dp-transformers/tree/main

Generate through baselines (icl, self-instruct, dp-gen)
```
python server/baselines_generate.py --dataset <Dataset Name> --alg icl --round 1000 --use_vllm
```

Post generation through professional model.
```
python server/post_generate.py --dataset <Dataset Name> --alg=post --base_model <Professional Model Path> --data_path=<Data Path From Last Step> --use_vllm 
```
