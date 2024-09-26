max_steps=10 # for dp
num_rounds=100
# batch_size=10 ###
batch_size=5 # for dp
gradient_accumulation_steps=1
seq_length=512 ###
num_clients=1 # for singleLLM
sample_clients=1
lora_r=16
lora_alpha=32   # twice of lora_r
# ['medalpaca/medical_meadow_medical_flashcards', 'FinGPT/fingpt-sentiment-train', "TIGER-Lab/MathInstruct", 'vicgalle/alpaca-gpt4', 'lucasmccabe-lmi/CodeAlpaca-20k', 'WizardLM/WizardLM_evol_instruct_70k', 'databricks/databricks-dolly-15k', 'tatsu-lab/alpaca']

dataset_name="new"

dataset_sample=500
lr=2e-5
gpu=7

# model_name="/mnt/workspace/wenhaowang/llama2"
# peft_model='/mnt/workspace/wenhaowang/EasyFedLLM/output/fingpt-sentiment-train_500_local0_c1s1_i10_b5a1_l512_r16a32_20240505152718/checkpoint-100'
        
# model_name="/mnt/workspace/wenhaowang/EasyFedLLM/output/fingpt-sentiment-train_500_local0_c1s1_i10_b5a1_l512_r16a32_20240505152718/full-100"
model_name="/ailab/user/wangwenhao/llama2"
# fed_alg="a"
# fed_alg="fedavg"
# fed_alg='dplocal0'
fed_alg="local0"
# fed_alg="dp"

output_dir=./output
CUDA_VISIBLE_DEVICES=$gpu python main_sft.py \
 --fedopt_eta 1e-3 \
 --fedopt_tau 1e-3 \
 --learning_rate $lr \
 --model_name $model_name \
 --dataset_name $dataset_name \
 --dataset_sample $dataset_sample \
 --fed_alg $fed_alg \
 --num_clients $num_clients \
 --sample_clients $sample_clients \
 --max_steps $max_steps \
 --num_rounds $num_rounds \
 --batch_size $batch_size \
 --gradient_accumulation_steps $gradient_accumulation_steps \
 --seq_length $seq_length \
 --peft_lora_r $lora_r \
 --peft_lora_alpha $lora_alpha \
 --use_peft \
 --output_dir $output_dir \
 --split_strategy not \
 --local_data_dir '/ailab/user/wangwenhao/KnowledgeSG/outputs/sampled_data/New/train/client0.json' \
 --load_in_8bit \
 --gradient_checkpoint True \
 --target_epsilon 20 \
 --per_sample_max_grad_norm 1.0 \
 #  --peft_model $peft_model \
#  --use_dp