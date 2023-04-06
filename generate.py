import sys

import peft.tuners.lora
import fire
import gradio as gr
from transformers import GenerationConfig

from utils import prompter
from utils.prompter import Prompter

assert peft.tuners.lora.is_gptq_available()

import torch
from autograd_4bit import load_llama_model_4bit_low_ram
from peft import PeftModel

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

from arg_parser import get_config


def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "tloen/alpaca-lora-7b",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = False,
):
    ft_config = get_config()
    # * Show loaded parameters
    if ft_config.local_rank == 0:
        print(f"{ft_config}\n")
    if ft_config.gradient_checkpointing:
        print('Disable Dropout.')
    # Load Basic Model
    model, tokenizer = load_llama_model_4bit_low_ram(ft_config.llama_q4_config_dir,
                                                     ft_config.llama_q4_model,
                                                     device_map=ft_config.device_map,
                                                     groupsize=ft_config.groupsize)
    model = PeftModel.from_pretrained(model, ft_config.lora_apply_dir, device_map={'': 0},
                                      torch_dtype=torch.float32)  # ! Direct copy from inference.py
    print(ft_config.lora_apply_dir, 'loaded')
    print('Fitting 4bit scales and zeros to half')
    for n, m in model.named_modules():
        if '4bit' in str(type(m)):
            if m.groupsize == -1:
                m.zeros = m.zeros.half()
            m.scales = m.scales.half()
    tokenizer.pad_token_id = 0
    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    prompter = Prompter(prompt_template)
    
    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return prompter.get_response(output)

    gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(
                lines=2,
                label="Instruction",
                placeholder="Tell me about alpacas.",
            ),
            gr.components.Textbox(lines=2, label="Input", placeholder="none"),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.1, label="Temperature"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.75, label="Top p"
            ),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            ),
            gr.components.Slider(
                minimum=1, maximum=4, step=1, value=4, label="Beams"
            ),
            gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
            ),
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=5,
                label="Output",
            )
        ],
        title="🦙",
        description=""
    ).launch(server_name="0.0.0.0", share=True)
    
if __name__ == "__main__":
    fire.Fire(main)
