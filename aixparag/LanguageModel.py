import os
from groq import Groq
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login
from openai import OpenAI
from pydantic import BaseModel, field_validator

class MessageInfo(BaseModel):
    tassonomia: list[str]
    macro_ambito: list[str]
    luogo: list[str]
    model_config = dict(extra="forbid")

    @field_validator('tassonomia', 'macro_ambito', 'luogo', mode='after')
    @classmethod
    def to_lower(cls, value):
        if isinstance(value, str):
            return value.lower()
        return value

class GroqModel:
    """
    A class to interact with the Groq API for generating text.
    It uses the 'llama-3.1-8b-instant' model and retrieves the API key
    from the 'GROQ_API_KEY' environment variable.
    """

    def __init__(self, model_name="llama-3.1-8b-instant"):
        """
        Initializes the Groq client.
        The API key is retrieved from the GROQ_API_KEY environment variable.
        Ensure this environment variable is set before running the code.
        """
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable not set. "
                "Please set it before initializing GroqModel."
            )
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def generate(self, sys_prompt: str, user_prompt: str) -> str:
        """
        Generates a response from the Groq API based on the given prompt.
        """
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": sys_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
                model=self.model_name,
                temperature=0.1
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"An error occurred during text generation: {e}")
            return f"Error: {e}"

class VLLMModel:
    """
    A class to interact with the vLLM API for generating text.
    It uses the 'meta-llama/Llama-3.1-8B-Instruct' model and retrieves the API key
    from the 'VLLM_API_KEY' environment variable.
    """
    def __init__(self):
        """
        Initializes the vLLM client.
        The API key is retrieved from the VLLM_API_KEY environment variable.
        Ensure this environment variable is set before running the code.
        """

        from start_api import start_api_openai_base_url, start_api_openai_base_model
        self.client = OpenAI(
        base_url = start_api_openai_base_url,
        # base_url = "http://localhost:1234/v1", # local vLLM endpoint
        api_key='ollama', # required, but unused
        )
        self.model_name = start_api_openai_base_model

    def generate(self, sys_prompt: str, conversation: list, max_new_tokens: int = 500, temperature: float = 0.9) -> str:
        if not conversation:
            return "Error: The conversation list cannot be empty."

        # The chat template expects a list of dictionaries with 'role' and 'content'
        messages = [{"role": "system", "content": sys_prompt}]
        for i, text in enumerate(conversation):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": text})

        try:
            # Apply the chat template and tokenize the input
            message = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_new_tokens
            ).choices[0].message.content
            return message

        except Exception as e:
            print(f"Error during reply generation: {e}")
            return "An error occurred while generating the reply."
        
    def generate_json(self, sys_prompt: str, conversation: list, max_new_tokens: int = 500, temperature: float = 0.9) -> str:
        if not conversation:
            return "Error: The conversation list cannot be empty."

        # The chat template expects a list of dictionaries with 'role' and 'content'
        messages = [{"role": "system", "content": sys_prompt}]
        for i, text in enumerate(conversation):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": text})

        try:
            message = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_new_tokens,
            response_format={
                        'type': 'json_schema',
                        'json_schema': {
                            'name' : 'message-info',
                            'schema' : MessageInfo.model_json_schema()
                        }
                    }
            ).choices[0].message.content
            return message

        except Exception as e:
            print(f"Error during reply generation: {e}")
            return "An error occurred while generating the reply."    

    
class HuggingFaceModel:
    """
    A class for interacting with a Hugging Face causal language model.

    Attributes:
        model_name (str): The name of the model to be loaded.
        device (str): The device to run the model on ('cuda' or 'cpu').
        tokenizer (AutoTokenizer): The tokenizer for the specified model.
        model (AutoModelForCausalLM): The loaded causal language model.
    """

    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        """
        Initializes the HuggingFaceModel with a specified model.

        Args:
            model_name (str): The name of the model to load from Hugging Face.
                              Defaults to "meta-llama/Llama-3.1-8B-Instruct".
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading model '{self.model_name}' on device '{self.device}'...")
        
        # Load the tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device
            )
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def generate(self, sys_prompt: str, conversation: list, max_new_tokens: int = 500, temperature: float = 0.9) -> str:
        """
        Generates a reply from the model based on a conversation history.

        The conversation is a list of alternating user and chatbot messages,
        starting with the user's query. The model will generate the next
        chatbot response.

        Args:
            conversation (list): A list of strings representing the conversation history.
                                 Example: ["User query 1", "Chatbot reply 1", "User query 2"]
            max_new_tokens (int): The maximum number of tokens to generate for the reply.
                                  Defaults to 100.

        Returns:
            str: The generated reply from the model.
        """
        if not conversation:
            return "Error: The conversation list cannot be empty."

        # The chat template expects a list of dictionaries with 'role' and 'content'
        messages = [{"role": "system", "content": sys_prompt}]
        for i, text in enumerate(conversation):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": text})

        try:
            # Apply the chat template and tokenize the input
            chat_template_applied = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.tokenizer(chat_template_applied, return_tensors="pt", truncation=True)
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            # Generate the response
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature = temperature
                )
            
            # Decode the generated output, skipping the original input tokens
            response = self.tokenizer.decode(
                outputs[0][input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            return response.strip()

        except Exception as e:
            print(f"Error during reply generation: {e}")
            return "An error occurred while generating the reply."


if __name__ == "__main__":
    model = VLLMModel()
    sys_prompt = "Generate a JSON with the fields tassonomia, macro_ambito and luogo. Use the following format: {\"tassonomia\": [..], \"macro_ambito\": [..], \"luogo\": [..]}. Only return the JSON, without any other text."
    conversation = ["Generate a JSON with the fields tassonomia, macro_ambito and luogo. Use the following format: {\"tassonomia\": [..], \"macro_ambito\": [..], \"luogo\": [..]}. Only return the JSON, without any other text."]
    reply = model.generate_json(sys_prompt, conversation)
    print(reply)