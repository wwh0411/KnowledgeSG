# KnowledgeSG: Privacy-Preserving Synthetic Text Generation With Knowledge Distillation From Server
This is the official repository for the EMNLP 2024 main conference paper "KnowledgeSG: Privacy-Preserving Synthetic Text Generation With Knowledge Distillation From Server".  
More code and illustration about the training, evaluation and visualization we used following previous works are on the way!
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

## Citation
```
@inproceedings{wang-etal-2024-knowledgesg,
    title = "{K}nowledge{SG}: Privacy-Preserving Synthetic Text Generation with Knowledge Distillation from Server",
    author = "Wang, WenHao  and
      Liang, Xiaoyu  and
      Ye, Rui  and
      Chai, Jingyi  and
      Chen, Siheng  and
      Wang, Yanfeng",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
}
```
