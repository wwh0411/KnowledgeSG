from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizerFast, LlamaForCausalLM
import datasets
import vllm
import sys
import json
import random
from tqdm import tqdm
import argparse
from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel


class Inferencer:
    def __init__(self, base_model_name=None, adapter_model_name=None, model=None, tokenizer=None, use_vllm=False):
        self.model_name = base_model_name
        self.use_vllm = use_vllm

        if use_vllm:
            self.vllm = LLM(
                model=self.model_name,
                trust_remote_code=True,
                tokenizer_mode='slow',
                # tensor_parallel_size=,
                # gpu_memory_utilization=1,
                max_model_len=2048,
            )
            self.sampling_params = SamplingParams(
                temperature=0.7,
                # do_sample=True,
                # repetition_penalty=1.1,  # 1.0 means 'off'. unfortunately if we penalize it it will not output Sphynx:
                # temperature=0.5,  # default: 1.0
                # top_k=50,  # default: 50
                # top_p=1.0,
                max_tokens=2048,
            )
        else:
            if model:  # 直接传model
                print("resume local model")
                self.model = model
            else:  # 加载local model
                self.model = LlamaForCausalLM.from_pretrained(base_model_name, device_map='auto', load_in_8bit=True)

            if adapter_model_name:
                self.model = PeftModel.from_pretrained(self.model, adapter_model_name)

        # about tokenizer
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            if base_model_name:
                self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
            else:
                self.tokenizer = None
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        self.eos_token = self.tokenizer.eos_token

    @staticmethod
    def extract_answer_instruct(prompt_output):

        result = prompt_output.split('###example:')[0]
        try:
            instruction = result.split('##instruction:')[1].strip()
        except:
            instruction = None

        return instruction

    @staticmethod
    def extract_answer_output(prompt_output):
        try:
            output = prompt_output.split('###example:')[0].split('##')[0].strip()
        except:
            output = None

        return output

    @staticmethod
    def extract_answer(prompt_output):

        result = prompt_output.split('###example:')[0]
        try:
            output = result.split('##output:')[1].strip()
            # 防止没有output的情形 / 把下一个的instruction放进来
            if '###example' in output or '### example' in output:
                output = None
            if '##instruction' in output or '##Instruction' in output:
                output = None
            instruction = result.split('##output:')[0].split('##instruction:')[1].strip()
        except:
            instruction, output = None, None
        return instruction, output

    def inference(self, prompt_input):
        # inference without vllm
        tokens = self.tokenizer.encode(prompt_input, return_tensors="pt").cuda()

        outputs = self.model.generate(inputs=tokens, max_length=1024,
                                      do_sample=True,
                                      repetition_penalty=1.1,
                                      # 1.0 means 'off'. unfortunately if we penalize it it will not output Sphynx:
                                      temperature=0.5,  # default: 1.0
                                      top_k=1,  # default: 50
                                      top_p=1.0,
            )
        prompt_output = self.tokenizer.decode(outputs[0])

        return prompt_output[len(prompt_input):]

    def inference_vllm(self, prompt_input):
        output = self.vllm.generate(prompt_input, self.sampling_params)
        if isinstance(prompt_input, list):
            prompt_output = [ou_.outputs[0].text for ou_ in output]
        else:
            prompt_output = output[0].outputs[0].text
        return prompt_output

