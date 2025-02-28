import google.generativeai as genai
from ..base import LLMProvider, LLMConfig
from ..key_manager import APIKeyManager
from typing import Optional, List, Iterator
import logging
import os
import warnings
from ...utils.text_splitter import TextSplitter
import tiktoken

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('grpc').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning)

from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)
absl_logging._warn_preinit_stderr = False

class GeminiConfig(LLMConfig):
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_keys: Optional[List[str]] = None,
        model: str = "gemini-2.0-flash",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.api_keys = api_keys
        self.model = model

class GeminiProvider(LLMProvider):
    def __init__(self, config: GeminiConfig):
        super().__init__(config)
        self.key_manager = APIKeyManager(config.api_keys)
        if config.api_key:
            self.key_manager.add_key(config.api_key)
        
        # Inicializa o splitter com contador de tokens
        self.splitter = TextSplitter(
            chunk_size=config.max_tokens,
            chunk_overlap=min(200, config.max_tokens // 5),  # 20% de overlap
            length_function=self._count_tokens
        )
            
        self._setup_provider()

    def _setup_provider(self):
        try:
            self.current_key = self.key_manager.get_next_key()
            if not self.current_key:
                raise ValueError("No API key provided")
                
            genai.configure(api_key=self.current_key)
            self._model = genai.GenerativeModel(self.config.model)
        except Exception as e:
            raise Exception(f"Failed to initialize Gemini: {str(e)}")

    def _count_tokens(self, text: str) -> int:
        """Conta tokens no texto usando tiktoken"""
        try:
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # Aproximação para Gemini
            return len(encoding.encode(text))
        except:
            # Fallback para estimativa simples
            return len(text.split())

    def generate(self, prompt: str, **kwargs) -> str:
        # Verifica se precisa dividir o prompt
        chunks = self.splitter.split_text(prompt)
        
        if len(chunks) == 1:
            # Se é um único chunk, processa normalmente
            return self._generate_single(prompt, **kwargs)
        
        # Processa múltiplos chunks
        responses = []
        context = ""
        
        for i, chunk in enumerate(chunks):
            if i > 0:
                # Adiciona contexto das respostas anteriores
                chunk = f"Previous context: {context}\n\nContinuing with: {chunk}"
            
            response = self._generate_single(chunk, **kwargs)
            responses.append(response)
            
            # Atualiza contexto para próximo chunk
            context = response[-200:]  # Mantém últimos 200 caracteres como contexto
        
        # Combina todas as respostas
        return "\n".join(responses)

    def _generate_single(self, prompt: str, **kwargs) -> str:
        try:
            generation_config = genai.GenerationConfig(
                max_output_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                temperature=kwargs.get('temperature', self.config.temperature),
            )
            
            # Formatar o prompt como o Gemini espera
            parts = [{"text": prompt}]
            
            print(f"Model: {self.config.model}")
            print(f"Temperature: {self.config.temperature}")
            print(f"Max Tokens: {self.config.max_tokens}")
            print(f"Prompt:\n{prompt[:200]}...")
            
            response = self._model.generate_content(
                parts,
                generation_config=generation_config
            )
            
            if hasattr(response, 'candidates'):
                if response.candidates:
                    candidate = response.candidates[0]
                    
            # Extrair texto da resposta
            if response and response.candidates:
                content = response.candidates[0].content.parts[0].text
                if content:
                    print("\nFinal Content:")
                    print(content[:200] + "..." if len(content) > 200 else content)
                    return content
                    
            raise Exception("Empty response from Gemini")
            
        except Exception as e:
            print(f"\nDEBUG - Error Details:")
            print(f"Error type: {type(e)}")
            print(f"Error message: {str(e)}")
            print(f"Full error: {e}")
            
            if "429" in str(e):
                self.key_manager.mark_key_failed(self.current_key)
                try:
                    self._try_with_new_key()
                    return self._generate_single(prompt, **kwargs)
                except ValueError as ve:
                    raise Exception("All API keys have failed or are rate limited. Please wait and try again.")
            raise Exception(f"Failed to generate content: {str(e)}")

    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """
        Implementação do streaming para Gemini.
        Como o Gemini não suporta streaming nativo, simulamos com generate
        """
        try:
            response = self.generate(prompt, **kwargs)
            yield response
        except Exception as e:
            raise Exception(f"Gemini streaming failed: {str(e)}")

    def _try_with_new_key(self):
        """Tenta configurar uma nova chave API"""
        try:
            self.current_key = self.key_manager.get_next_key()
            if not self.current_key:
                raise ValueError("No available API keys")
            genai.configure(api_key=self.current_key)
            self._model = genai.GenerativeModel(self.config.model)
        except Exception as e:
            raise Exception(f"Failed to switch API key: {str(e)}")

    def __del__(self):
        """Cleanup quando o provider é destruído"""
        try:
            # Limpar recursos do gRPC de forma silenciosa
            import grpc
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                grpc.shutdown_all_channels()
        except:
            pass 