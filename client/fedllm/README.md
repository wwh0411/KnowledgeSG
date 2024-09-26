## TODO

- [ ] Check the issue of dataset loading 'loading cached processed ...'
- [ ] Vicuna eval: Different temperature for different category.
- [ ] Unify training template with Fastchat.
- [ ] Learning rate warmup

## Structure

- `main_sft.py` contains the SFT training code / `run_sft.sh` is the running script.
- `main_dpo.py` contains the DPO training code / `run_dpo.sh` is the running script.
- `config.py` contains all the arguments related to setting and training.
- `/fed` directory contains FL algorithms and operations related to FL.
- `/utils` directory contains general operations.
- `/eval` directory contains codes related to open-ended evaluation (e.g., alpaca-eval, vicuna bencn, mt-bench), GPT as the judger.

## Supported

**Algorithms:** Central, Local, FedAvg, FedProx, FedAvgM, FedAdagrad, FedYogi, FedAdam

```
Run client 0: --fed_alg local0 --num_clients x
Run central: --fed_alg fedavg --num_clients 1
Run FL: --fed_alg xx --num_client xx
```

**Learning Rate:** Mannual learning rate schedule at each round. Being constant during local training.  # TODO: warmup

**Datasets:**


## Training

### Federated SFT




### Federated DPO

- Currently support two datasets: HH-RLHF and UltraFeedback
- Learning rate: 5e-5 for UltraFeedback, 5e-4 for HH-RLHF
- Use the conversation template from FastChat
- !!! Remember to use the **corresponding template** of your SFT model. Put it into `MODEL2TEMPLATE` in `utils/template.py`.


## Evaluation

### Open-Ended

**(Optional) Merge LoRA:** 

- Needed if you want to use vllm.
- <checkpoint_path>: base_path/checkpoint-xx, then the full model will be saved at base_path/full-xx (i.e., <full_path>)
```
python utils/merge_lora.py --lora-path <checkpoint_path>
```

**Generate Answers:**

- Put [`vicuna.jsonl`, `mtbench.jsonl`] in `./eval/eval_data`. Download [Here](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/data/).
- (Optional) Use vLLM for efficient inference. 

```
# See run_generate_answer.sh

python generate_answer.py --use_vllm --model_name <full_path> --benchmark_name <vicuna or alpaca>   # for vicuna and alpaca

python generate_answer_mt.py --lora_path <lora_path> --base_model_path ehartford/Wizard-Vicuna-7B-Uncensored   # for mt bench
```

**GPT as Judger:**
- Create a `gpt.yaml` file under `eval/`. Define the following keys:
```
api_key: "xx"
api_type: "azure"
api_base: "https://xx"
api_version: "2023-05-15"
engine: "gpt-4"
```
- For Alpaca and Vicuna benchmark. See `run_judge.sh` for two evaluation protocols
```
# First is fastchat eval (single judge).
python judge_vicuna.py --model_outputs $model_outputs --judger $judger 

# Second is alpaca eval (pariwise judge). This has the issue of position bias. Suggest run two parallel evaluations by exchanging ref_position
python judge_alpaca.py --model_outputs $model_outputs --model_outputs_ref $model_outputs_ref --judger $judger --ref_position $ref_position
```

- For MT Bench evaluation, we currently use the codes of [FastChat](https://github.com/lm-sys/FastChat)

```
- export OPENAI_API_KEY=[YOUR_API_KEY_HERE]

- Put the generated <answer_name>.jsonl under FastChat/fastchat/llm_judge/data/mt_bench/model_answer/

- Under `FastChat/fastchat/llm_judge/`: python gen_judgment.py --judge-model gpt-4-1106-preview --model-list <answer_name>

- Show results: python show_result.py --judge-model gpt-4-1106-preview --model-list <answer_name_1> <answer_name_2>
```

    

### Close-Ended
This repo consists several metrics, including [MMLU BBH DROP HumanEval CRASS]. https://github.com/declare-lab/instruct-eval

**Setup Steps:**
```
conda create -n instruct-eval python=3.8 -y
conda activate instruct-eval
git clone https://github.com/declare-lab/instruct-eval
cd instruct-eval
pip install -r requirements.txt
pip install datasets --upgrade      # for BBH
mkdir -p data
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar -O data/mmlu.tar
tar -xf data/mmlu.tar -C data && mv data/data data/mmlu
```

**Scripts:**
```
# Note!!! this repo just prints results rather than save them.
python main.py mmlu --model_name llama --model_path meta-llama/Llama-2-7b-hf --lora_path <your_saved_lora_dir>
python main.py all --model_name llama --model_path meta-llama/Llama-2-7b-hf --lora_path <your_saved_lora_dir>   # [MMLU BBH DROP HumanEval CRASS]
```
