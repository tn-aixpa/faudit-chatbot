import copy
import json
from typing import List, Dict
from pprint import pprint
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel



DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct" # "gpt-4o-mini-2024-07-18"  #gpt-3.5-turbo-0125"  "gpt-4o-mini-2024-07-18" "gpt-3.5-turbo-0125"
DEFAULT_TEMPERATURE = 0.5 # 0.7
DEFAULT_MAX_TOKENS = 250 # 500
ADAPTER_PATH = ""
HF_TOKEN = ""
DEFAULT_OPTIONS_NUMBER = 3

# Load model directly
tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL)

class GenerationLlamaAdapter:
    def __init__(self,
                 model_name = DEFAULT_MODEL,
                 adapter_path = ADAPTER_PATH,
                 temperature = DEFAULT_TEMPERATURE,
                 max_tokens = DEFAULT_MAX_TOKENS,
                 hf_token = HF_TOKEN,
                 question_sys_prompt = "",
                 options_number = DEFAULT_OPTIONS_NUMBER):

        self.model_name = model_name
        self.adapter_path = adapter_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.hf_token = hf_token
        self.question_sys_prompt = question_sys_prompt
        self.options_number = options_number 

        self.bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
        # Load the base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.bnb_config,
            device_map="auto",  # Automatically map layers to available devices (e.g., GPU/CPU)
            use_auth_token=self.hf_token
        )

        # Wrap the model with PEFT to load the adapter
        self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_auth_token=self.hf_token)


    def generate_text(self, input_chat, nr_gen):
        # Generate the next turn from the prompt string
        if input_chat is None or not isinstance(input_chat, list) or len(input_chat) == 0:
            raise ValueError("input_chat is None or empty. Ensure valid input before tokenization.")

        input_chat = self.tokenizer.apply_chat_template(input_chat, tokenize=False)
        inputs = self.tokenizer(input_chat, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_tokens,  # Limit the number of tokens in the response
            do_sample=True,  # Enable sampling for variability
            temperature=self.temperature,  # Sampling temperature to control randomness
            num_return_sequences=nr_gen,
            top_p=0.9,
            # # top_k=100,
            # repetition_penalty=2.0
        )

        # Decode and print the generated text
        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # response = response.split("assistant")[-1]
        return response



