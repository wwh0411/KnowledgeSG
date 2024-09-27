import datasets
import random

class Prompter:
    def __init__(self, ):
        pass

    def gen_sample_list(self, dataset_hf, index_list=None):
        # random select 3
        # dataset_hf = self.select_code_sample(dataset_hf)
        if index_list == None:
            index_list = random.sample(range(0, len(dataset_hf)), 3)

        # select
        dataset_sample = dataset_hf.select(index_list)
        return dataset_sample, index_list

    def map_function(self, dataset_sample):
        # gen few shot
        # template_sample = "###example:\n\n##instruction: \n{instruction} {input}\n##output:\n{output}\n\n"
        template_sample = "###example:\n\n##instruction: \n{instruction} \n##output:\n{output}\n\n"

        fewshot = ""
        for sample in dataset_sample:
            instruction, output = sample['instruction'], sample['response']
            # output += self.eos_token
            prompt_sample = template_sample.format(instruction=instruction, output=output)
            # prompt_sample += self.eos_token
            fewshot += prompt_sample

        # gen full prompt
        template = "Based on the following examples, please generate a new and unique example that is different and follows the underlying pattern or theme. Try to make your generation as diverse as possible.\n\n{fewshot}###example:"
        # template = "Based on the following examples, generate a new example about code writing\n\n{fewshot}###example:"
        # template = "Based on the following five examples, please generate only one new and unique example that is different and diverse. Try to make your generation as diverse as possible.\n\n{fewshot}###new example:"
        # template = "Coming up with a serious of examples.\n\n{fewshot}###example:"
        prompt = template.format(fewshot=fewshot)

        return prompt

    def map_self_instruct_1(self, dataset):
        template_sample = '###example:\n\n##instruction: {instruction}\n'
        fewshot = ""
        for sample in dataset:
            instruction, output = sample['instruction'], sample['response'],

            prompt_sample = template_sample.format(instruction=instruction)
            fewshot += prompt_sample

        template = "{fewshot}###example:"
        prompt = template.format(fewshot=fewshot)
        return prompt

    def map_self_instruct_2(self, dataset, new_instruction, icl=False):
        template_system = "Come up with examples for the following tasks. Try to generate multiple examples when possible. If the task doesn't require additional input, you can generate the output directly.\n{fewshot}\n###example:\n\n##instruction: \n{new_instruction}\n##output:"
        template_sample = "###example:\n\n##instruction: \n{instruction}\n##output:\n{output}\n\n"
        fewshot = ""
        if icl:
            for sample in dataset:
                instruction, output = sample['instruction'], sample['response'],
                prompt_sample = template_sample.format(instruction=instruction, output=output)
                fewshot += prompt_sample

        prompt = template_system.format(fewshot=fewshot, new_instruction=new_instruction)
        return prompt

    def map_instruct(self, instruction):
        template = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
        return template.format(instruction=instruction)

    def map_fin(self, instruction):
        return "Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}\n" + instruction + "\nAnswer: "

    def map_med(self, instruction):
        return "If you are a doctor, please answer the medical questions based on the patient's description. \n\nPatient: " + instruction + "\nChatDoctor: "

    def create_multi_prompts(self, dataset_hf, max_samples):
        prompt_list = []
        for i in range(max_samples):
            prompt = self.map_function(dataset_hf)
            prompt_list.append(prompt)
        return prompt_list
