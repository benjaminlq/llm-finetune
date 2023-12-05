import os
import tiktoken

from transformers.tokenization_utils import PreTrainedTokenizer
from typing import Optional, List, Union, Callable
from datasets import Dataset, DatasetDict

def create_llama2_chat_prompt(
    messages: List[str], tokenizer: PreTrainedTokenizer, system_prompt: Optional[str] = None
) -> str:
    prompt_str = ""

    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS_TOKEN, EOS_TOKEN = tokenizer.bos_token, tokenizer.eos_token

    if system_prompt:
        messages[0] = B_SYS + system_prompt + E_SYS + messages[0]
        
    for idx, (user_message, assistant_message) in enumerate(zip(messages[::2], messages[1::2])):
        if idx == 0 and tokenizer.add_bos_token:
            user_message = B_INST + " " + user_message + " " + E_INST
        else:
            user_message = BOS_TOKEN + " " + B_INST + " " + user_message + " " + E_INST
        assistant_message = " " + assistant_message + " " + EOS_TOKEN
        
        prompt_str += user_message
        prompt_str += assistant_message

    if (len(messages) % 2) == 1:       
        user_query = BOS_TOKEN + " " + B_INST + " " + messages[-1] + " " + E_INST
        prompt_str += user_query
    
    return prompt_str.strip()

ALPACA_INSTRUCTION_COMPLETION_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

ALPACA_INSTRUCTION_TEMPLATE_NO_INPUT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

def create_llama2_instruction_prompt(
    instruction: str,
    input: Optional[str] = None,
    output: Optional[str] = None,
    prompt_template: str = ALPACA_INSTRUCTION_COMPLETION_TEMPLATE
) -> str:
    if not input:
        prompt_template = ALPACA_INSTRUCTION_TEMPLATE_NO_INPUT
        prompt_str = prompt_template.format(instruction = instruction)
        
    else:
        prompt_str = prompt_template.format(instruction=instruction, input=input)
    
    if output:
        prompt_str += f"\n{output}"
            
    return prompt_str

def save_dataset_to_json(
    datasets: Union[Dataset, DatasetDict],
    output_directory
):
    if isinstance(datasets, Dataset):
        datasets = DatasetDict(data=datasets)
    for dataset_name, dataset in datasets.items():
        dataset.to_json(os.path.join(output_directory, f"{dataset_name}.json"))
        
def count_tokens(
    text: str,
    tokenizer: Callable = tiktoken.encoding_for_model("gpt-3.5-turbo")
):
    return len(tokenizer.encode(text))