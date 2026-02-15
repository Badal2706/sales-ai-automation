"""
Follow-up generation module.
Creates contextual email and WhatsApp messages using local LLM with GPU.
"""

import requests
from typing import Tuple

from config import LLM_CONFIG, check_gpu_availability
from models import FollowUpContent
from prompts import Prompts
from memory import MemoryManager

class FollowUpGenerator:
    """
    Generates personalized follow-up communications.
    Uses client history for contextual relevance.
    """

    def __init__(self, memory: MemoryManager = None):
        self.config = LLM_CONFIG
        self.provider = self.config['provider']
        self.memory = memory or MemoryManager()
        self.gpu_config = self._setup_gpu()

    def _setup_gpu(self) -> dict:
        """Configure GPU settings."""
        gpu_info = check_gpu_availability()
        return {
            'use_gpu': self.config.get('gpu', False) and gpu_info['available'],
            'type': gpu_info['type']
        }

    def _call_llm(self, prompt: str) -> str:
        """Route to appropriate LLM provider."""
        if self.provider == 'ollama':
            return self._call_ollama(prompt)
        elif self.provider == 'transformers':
            return self._call_transformers(prompt)
        elif self.provider == 'llama_cpp':
            return self._call_llama_cpp(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API."""
        try:
            options = {
                'temperature': 0.7,  # Slightly higher for creativity
                'num_predict': self.config['max_tokens']
            }

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
            return response.json()['response'].strip()
        except Exception as e:
            raise RuntimeError(f"Follow-up generation failed: {str(e)}")

    def _call_transformers(self, prompt: str) -> str:
        """HuggingFace transformers with GPU."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if not hasattr(self, '_generator'):
                print(f"🚀 Loading follow-up model on {'GPU' if self.gpu_config['use_gpu'] else 'CPU'}...")

                model_name = self.config['model']

                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._model = AutoModelForCausalLM.from_pretrained(model_name)

                if self.gpu_config['use_gpu']:
                    self._model = self._model.to(f"cuda:{self.config.get('cuda_device', 0)}")

                self._generator = pipeline(
                    'text-generation',
                    model=self._model,
                    tokenizer=self._tokenizer,
                    max_new_tokens=self.config['max_tokens'],
                    temperature=0.7,
                    device=0 if self.gpu_config['use_gpu'] else -1
                )

            result = self._generator(prompt, return_full_text=False)
            return result[0]['generated_text'].strip()
        except Exception as e:
            raise RuntimeError(f"Transformers error: {str(e)}")

    def _call_llama_cpp(self, prompt: str) -> str:
        """llama.cpp with GPU."""
        try:
            from llama_cpp import Llama

            if not hasattr(self, '_llm'):
                kwargs = {
                    'model_path': self.config['model_path'],
                    'n_ctx': self.config.get('n_ctx', 4096),
                    'verbose': False
                }

                if self.gpu_config['use_gpu']:
                    kwargs['n_gpu_layers'] = self.config.get('n_gpu_layers', -1)

                self._llm = Llama(**kwargs)

            output = self._llm(
                prompt,
                max_tokens=self.config['max_tokens'],
                temperature=0.7,
                stop=["</s>"]
            )

            return output['choices'][0]['text'].strip()
        except Exception as e:
            raise RuntimeError(f"llama.cpp error: {str(e)}")

    def generate(self, client_id: int, crm_data: dict) -> FollowUpContent:
        """
        Generate both email and message follow-ups.

        Args:
            client_id: Client ID for context retrieval
            crm_data: Dict with summary, deal_stage, interest_level, etc.
        """
        # Get client context
        client = self.memory.db.get_client(client_id)
        if not client:
            raise ValueError(f"Client {client_id} not found")

        history = self.memory.get_context_for_ai(client_id)

        # Extract data with defaults
        summary = crm_data.get('summary', '')
        deal_stage = crm_data.get('deal_stage', 'prospecting')
        interest_level = crm_data.get('interest_level', 'neutral')
        next_action = crm_data.get('next_action', 'Follow up')
        objections = crm_data.get('objections')  # Can be None

        # Generate email
        email_prompt = Prompts.get_email_prompt(
            client_name=client.name,
            company=client.company,
            history=history,
            summary=summary,
            deal_stage=deal_stage,
            interest_level=interest_level,
            next_action=next_action,
            objections=objections
        )

        email_text = self._call_llm(email_prompt)

        # Generate WhatsApp message
        message_prompt = Prompts.get_message_prompt(
            client_name=client.name,
            summary=summary,
            next_action=next_action,
            interest_level=interest_level
        )

        message_text = self._call_llm(message_prompt)

        return FollowUpContent(
            email_text=email_text,
            message_text=message_text
        )

def generate_followups(client_id: int, crm_data: dict) -> FollowUpContent:
    """
    Convenience function for follow-up generation.
    """
    generator = FollowUpGenerator()
    return generator.generate(client_id, crm_data)