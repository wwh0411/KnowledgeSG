from utils.utils import *
from tqdm import tqdm


def check_num_for_vllm(outputs, inputs):
    if len(outputs) != len(inputs):
        print(len(inputs), len(outputs))
        i = 0
        while i < len(outputs) - 1:
            if outputs[i] == '' and outputs[i + 1] == '':
                del outputs[i]
            i += 1
        if len(outputs) != len(inputs):
            raise ValueError('not equal list size')
    return outputs


def filter_instructions_by_similarity(inferencer, outputs, dataset_sample_list):
    instruction_list = []
    # extract answer
    for i, prompt_output in enumerate(outputs):

        instruction = inferencer.extract_answer_instruct(prompt_output)
        # judge if instruction/output is None
        if not instruction:
            continue

        # calc similarity
        sign = 0
        for sample in dataset_sample_list[i]:
            # for sample in dataset_sample:
            instruction_o = sample['instruction']

            try:
                sim_ = calculate_similarity(instruction, instruction_o)
                # filter by similarity # important
                if sim_ > 0.52:
                    sign = 1
            except:
                sign = 1
                continue

        if sign == 0 and len(instruction) > 1:
            instruction_list.append(instruction)
    return instruction_list


def generation(prompt_list, inferencer, use_vllm=False):

    # 生成
    if use_vllm:
        outputs = inferencer.inference_vllm(prompt_list)
    else:
        outputs = [inferencer.inference(prompt) for prompt in tqdm(prompt_list, desc='Inferencing:')]

    outputs = check_num_for_vllm(outputs, prompt_list)
    return outputs