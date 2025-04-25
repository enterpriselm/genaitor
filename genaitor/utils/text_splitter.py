from typing import List, Optional
import re

class TextSplitter:
    def __init__(
        self, 
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: callable = len,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.separators = separators or ["\n\n", "\n", ". ", ", ", " ", ""]
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks while preserving context"""
        # Se o texto é menor que o tamanho do chunk, retorna como está
        if self.length_function(text) <= self.chunk_size:
            return [text]
            
        chunks = []
        
        # Tenta diferentes separadores até encontrar um que funcione
        for separator in self.separators:
            if separator == "":
                # Último recurso: divide por caracteres
                return self._split_by_chars(text)
                
            segments = text.split(separator)
            
            # Se a divisão resultar em segmentos muito grandes, tenta próximo separador
            if any(self.length_function(seg) > self.chunk_size for seg in segments):
                continue
                
            # Combina segmentos em chunks
            current_chunk = []
            current_length = 0
            
            for segment in segments:
                segment_len = self.length_function(segment)
                
                if current_length + segment_len > self.chunk_size:
                    # Salva chunk atual e começa novo
                    if current_chunk:
                        chunks.append(separator.join(current_chunk))
                    current_chunk = [segment]
                    current_length = segment_len
                else:
                    current_chunk.append(segment)
                    current_length += segment_len
            
            # Adiciona último chunk
            if current_chunk:
                chunks.append(separator.join(current_chunk))
            
            # Se conseguiu dividir, retorna os chunks
            if chunks:
                return self._add_overlap(chunks)
        
        # Se nenhum separador funcionou, divide por caracteres
        return self._split_by_chars(text)
    
    def _split_by_chars(self, text: str) -> List[str]:
        """Divide texto por caracteres quando nenhum separador funciona"""
        return [
            text[i:i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
        ]
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Adiciona sobreposição entre chunks para manter contexto"""
        if len(chunks) <= 1:
            return chunks
            
        result = []
        for i in range(len(chunks)):
            if i == 0:
                result.append(chunks[i])
            else:
                # Adiciona parte do chunk anterior para contexto
                prev_chunk = chunks[i-1]
                overlap_size = min(self.chunk_overlap, len(prev_chunk))
                overlap = prev_chunk[-overlap_size:]
                result.append(overlap + chunks[i])
        
        return result 