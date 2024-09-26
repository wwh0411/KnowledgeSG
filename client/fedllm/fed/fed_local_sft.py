import torch
import copy
from trl import SFTTrainer
from transformers import TrainerCallback
from peft import get_peft_model_state_dict, set_peft_model_state_dict
import dp_transformers
import transformers
from .fed_local_dp import OpacusDPTrainer

def get_fed_local_sft_trainer(script_args, fed_args, model, tokenizer, training_args, privacy_args, local_dataset, formatting_prompts_func, data_collator, global_dict, local_auxiliary, global_auxiliary, round=0):
    
    if fed_args.fed_alg == 'fedprox':
        trainer = SFTTrainerFedProx(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
            global_state=global_dict,
            prox_mu=fed_args.prox_mu,
        )
    elif fed_args.fed_alg == 'scaffold':
        trainer = SFTTrainerSCAFFOLD(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
            global_state=global_dict,
            local_auxiliary=local_auxiliary,
            global_auxiliary=global_auxiliary,
        )
    elif fed_args.fed_alg in ['fedavg', 'a'] or (fed_args.fed_alg).startswith('local'):
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
        )
    elif 'test' in fed_args.fed_alg:
        print('test trainer ===')
        trainer = dp_transformers.dp_utils.OpacusDPTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
            privacy_args=privacy_args,
        )
    elif fed_args.fed_alg == 'dp' or (fed_args.fed_alg).startswith('dplocal'):
        # data_collator = dp_transformers.DataCollatorForPrivateCausalLanguageModeling(tokenizer)
        
        # import sys
        # sys.path.append("..")
        
        from utils.template import reformat
        local_dataset = reformat(local_dataset, tokenizer.eos_token)
        # print(local_dataset[0])
        local_dataset = local_dataset.map(
                lambda batch: tokenizer(batch['ins+res'], padding=True, truncation=True, max_length=script_args.seq_length),
                batched=True, num_proc=1, desc="tokenizing dataset", remove_columns=['instruction', 'response', 'ins+res']
            )
        local_dataset = local_dataset.map(lambda example:{"labels": example["input_ids"]}, batched=True)
        # print(local_dataset[0])
        
        # trainer = dp_transformers.dp_utils.OpacusDPTrainer(
        trainer = OpacusDPTrainer(
            round=round,
            args=training_args,
            model=model,
            # max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            eval_dataset=local_dataset,
            # formatting_func=formatting_prompts_func,
            data_collator=data_collator,
            privacy_args=privacy_args,
        )
        


    return trainer

class SFTTrainerFedProx(SFTTrainer):
    def __init__(self, global_state, prox_mu, **kwargs):
        super(SFTTrainerFedProx, self).__init__(**kwargs)
        self.global_state = global_state
        self.mu = prox_mu
    
    def compute_loss(self, model, inputs, return_outputs=False):

        return_values = super(SFTTrainerFedProx, self).compute_loss(model, inputs, return_outputs=return_outputs)

        if return_outputs:
            loss, outputs = return_values
        else:
            loss = return_values

        # Apply FedProx Loss
        for name, param in model.named_parameters():
            name = name.replace(".default", "")     # TODO: May need changes. to accord with peft
            # only trainable parameters
            if not param.requires_grad:
                continue
            else:
                loss += self.mu / 2 * torch.norm(param - self.global_state[name]) ** 2

        return (loss, outputs) if return_outputs else loss


class SFTTrainerSCAFFOLD(SFTTrainer):
    def __init__(self, global_state, local_auxiliary, global_auxiliary, **kwargs):
        super(SFTTrainerSCAFFOLD, self).__init__(**kwargs)
        self.global_state = global_state
        self.local_auxiliary = local_auxiliary
        self.global_auxiliary = global_auxiliary
        self.correction = copy.deepcopy(local_auxiliary)

        # for name, param in self.correction.items():
        #     # param.requires_grad = False
        #     param = self.global_auxiliary[name] - self.local_auxiliary[name]
        for name in self.correction.keys():
            self.correction[name] = self.global_auxiliary[name] - self.local_auxiliary[name]
    
    def get_auxiliary_param(self):
        auxiliary_new_para = copy.deepcopy(self.local_auxiliary)
        auxiliary_delta_para = copy.deepcopy(self.local_auxiliary)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                else:
                    name = name.replace(".default", "")
                    auxiliary_new_para[name] = (self.global_state[name] - param) / (self.args.max_steps * self.args.learning_rate) - self.correction[name]
                    auxiliary_delta_para[name] = auxiliary_new_para[name] - self.local_auxiliary[name]
        # print("1. Updated local auxiliary parameters")
        return auxiliary_new_para, auxiliary_delta_para

class SCAFFOLD_Callback(TrainerCallback):
    def __init__(self, correction, model):
        super(SCAFFOLD_Callback, self).__init__()
        self.correction = correction
        self.model = model
    def on_step_end(self, args, state, control, **kwargs):
        model_para = copy.deepcopy(get_peft_model_state_dict(self.model))
        for name in model_para.keys():
            model_para[name] -= args.learning_rate * self.correction[name]
        set_peft_model_state_dict(self.model, model_para)