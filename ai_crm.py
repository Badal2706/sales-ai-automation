"""
CRM AI module for extracting structured data from conversations.
Handles local LLM integration with GPU acceleration support.
"""

import json
import re
from typing import Optional, Union
import requests
import os

from config import LLM_CONFIG, check_gpu_availability
from models import CRMData, validate_json_output
from prompts import Prompts

class CRMExtractor:
    """
    Extracts structured CRM data from raw conversation text.
    Uses local LLM (Ollama, transformers, or llama.cpp) with GPU support.
    """

    def __init__(self):
        self.config = LLM_CONFIG
        self.provider = self.config['provider']
        self.gpu_config = self._setup_gpu()

        if self.provider == 'ollama':
            self._check_ollama()

    def _setup_gpu(self) -> dict:
        """Configure GPU settings based on availability."""
        gpu_info = check_gpu_availability()
        config = {
            'use_gpu': self.config.get('gpu', False) and gpu_info['available'],
            'type': gpu_info['type'],
            'device': self.config.get('cuda_device', 0)
        }
        return config

    def _check_ollama(self) -> None:
        """Verify Ollama is running and configure GPU."""
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=5)
            if response.status_code != 200:
                raise ConnectionError("Ollama not responding")

            # Ollama automatically uses GPU if available and model supports it
            # No additional configuration needed, but we can verify GPU is being used
            if self.gpu_config['use_gpu']:
                print(f"🚀 Ollama will use GPU ({self.gpu_config['type']}) if model supports it")

        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Ollama not running. Start with: ollama serve"
            )

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API with optimized settings."""
        try:
            # Ollama automatically uses GPU, but we can set num_gpu
            options = {
                'temperature': self.config['temperature'],
                'num_predict': self.config['max_tokens']
            }

            # If specific GPU layers configured
            if 'gpu_layers' in self.config and self.config['gpu_layers'] != -1:
                options['num_gpu'] = self.config['gpu_layers']

            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': self.config['model'],
                    'prompt': prompt,
                    'stream': False,
                    'options': options
                },
                timeout=self.config['timeout']
            )
            response.raise_for_status()
            return response.json()['response']
        except requests.exceptions.Timeout:
            raise TimeoutError("Ollama request timed out. Model may be loading.")
        except Exception as e:
            raise RuntimeError(f"Ollama error: {str(e)}")

    def _call_transformers(self, prompt: str) -> str:
        """
        HuggingFace transformers with GPU/quantization support.
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            # Lazy load model on first use
            if not hasattr(self, '_model'):
                print(f"🚀 Loading model on {'GPU' if self.gpu_config['use_gpu'] else 'CPU'}...")

                model_name = self.config['model']

                # Quantization config for GPU memory efficiency
                quantization = self.config.get('quantization')
                bnb_config = None

                if quantization and self.gpu_config['use_gpu']:
                    if quantization == "4bit":
                        bnb_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16
                        )
                    elif quantization == "8bit":
                        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

                # Device map for multi-GPU
                device_map = "auto" if self.gpu_config['use_gpu'] else "cpu"

                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map=device_map,
                    torch_dtype=torch.float16 if self.gpu_config['use_gpu'] else torch.float32
                )

                if self.gpu_config['use_gpu']:
                    print(f"✅ Model loaded on GPU ({self.gpu_config['type']})")

            # Tokenize and generate
            inputs = self._tokenizer(prompt, return_tensors="pt")

            if self.gpu_config['use_gpu']:
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.config['max_tokens'],
                temperature=self.config['temperature'],
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id
            )

            result = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove the input prompt from output
            if prompt in result:
                result = result.replace(prompt, "").strip()

            return result

        except ImportError:
            raise ImportError(
                "Transformers not installed. Install with: pip install transformers accelerate bitsandbytes"
            )
        except Exception as e:
            raise RuntimeError(f"Transformers error: {str(e)}")

    def _call_llama_cpp(self, prompt: str) -> str:
        """
        llama.cpp with GPU acceleration (fastest local option).
        """
        try:
            from llama_cpp import Llama

            # Lazy load
            if not hasattr(self, '_llm'):
                print(f"🚀 Loading llama.cpp model on {'GPU' if self.gpu_config['use_gpu'] else 'CPU'}...")

                model_path = self.config['model_path']
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model not found: {model_path}")

                kwargs = {
                    'model_path': model_path,
                    'n_ctx': self.config.get('n_ctx', 4096),
                    'verbose': False
                }

                if self.gpu_config['use_gpu']:
                    kwargs['n_gpu_layers'] = self.config.get('n_gpu_layers', -1)
                    print(f"✅ Offloading {kwargs['n_gpu_layers']} layers to GPU")

                self._llm = Llama(**kwargs)

            output = self._llm(
                prompt,
                max_tokens=self.config['max_tokens'],
                temperature=self.config['temperature'],
                stop=["</s>", "User:", "Human:"]
            )

            return output['choices'][0]['text'].strip()

        except ImportError:
            raise ImportError("Install llama-cpp-python with: pip install llama-cpp-python")
        except Exception as e:
            raise RuntimeError(f"llama.cpp error: {str(e)}")

    def _clean_json_response(self, raw: str) -> str:
        """
        Clean LLM output to extract valid JSON.
        Handles markdown code blocks and extra text.
        """
        # Remove markdown code blocks
        if '```json' in raw:
            raw = raw.split('```json')[1].split('```')[0]
        elif '```' in raw:
            raw = raw.split('```')[1].split('```')[0]

        # Find JSON object
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return match.group(0)

        return raw.strip()

    def extract(self, conversation: str, context: str = "New client") -> CRMData:
        """
        Main extraction method.
        Takes raw conversation, returns structured CRM data.
        """
        # Build prompt
        prompt = Prompts.get_crm_prompt(conversation, context)

        # Route to appropriate LLM
        if self.provider == 'ollama':
            raw_output = self._call_ollama(prompt)
        elif self.provider == 'transformers':
            raw_output = self._call_transformers(prompt)
        elif self.provider == 'llama_cpp':
            raw_output = self._call_llama_cpp(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        # Clean and validate
        cleaned = self._clean_json_response(raw_output)

        try:
            return validate_json_output(cleaned)
        except ValueError as e:
            # Retry once with stricter prompt if validation fails
            retry_prompt = prompt + "\n\nCRITICAL: Return ONLY valid JSON. No other text."

            if self.provider == 'ollama':
                raw_output = self._call_ollama(retry_prompt)
            elif self.provider == 'transformers':
                raw_output = self._call_transformers(retry_prompt)
            else:
                raw_output = self._call_llama_cpp(retry_prompt)

            cleaned = self._clean_json_response(raw_output)
            return validate_json_output(cleaned)

def extract_crm_data(conversation: str, context: str = "New client") -> CRMData:
    """
    Convenience function for CRM extraction.
    """
    extractor = CRMExtractor()
    return extractor.extract(conversation, context)