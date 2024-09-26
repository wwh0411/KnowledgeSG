checkpoint=200
base_path=../output
# base_model_name=ehartford/Wizard-Vicuna-7B-Uncensored
base_model_name=/GPFS/public/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9

for value in CodeAlpaca-20k_20000_fedavg_c10s2_i10_b8a2_l1024_20231129151258 CodeAlpaca-20k_20000_local0_c10s2_i10_b8a2_l1024_20231129151243
do
    exp_path=$base_path/$value/checkpoint-$checkpoint
    python merge_lora.py --lora-path $exp_path --base-model-name $base_model_name
done