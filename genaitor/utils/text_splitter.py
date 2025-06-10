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
        if self.length_function(text) <= self.chunk_size:
            return [text]
            
        chunks = []
        
        for separator in self.separators:
            if separator == "":
                return self._split_by_chars(text)
                
            segments = text.split(separator)
            
            if any(self.length_function(seg) > self.chunk_size for seg in segments):
                continue
                
            current_chunk = []
            current_length = 0
            
            for segment in segments:
                segment_len = self.length_function(segment)
                
                if current_length + segment_len > self.chunk_size:
                    if current_chunk:
                        chunks.append(separator.join(current_chunk))
                    current_chunk = [segment]
                    current_length = segment_len
                else:
                    current_chunk.append(segment)
                    current_length += segment_len
            
            if current_chunk:
                chunks.append(separator.join(current_chunk))
            
            if chunks:
                return self._add_overlap(chunks)
        
        return self._split_by_chars(text)
    
    def _split_by_chars(self, text: str) -> List[str]:
        return [
            text[i:i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
        ]
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        if len(chunks) <= 1:
            return chunks
            
        result = []
        for i in range(len(chunks)):
            if i == 0:
                result.append(chunks[i])
            else:
                prev_chunk = chunks[i-1]
                overlap_size = min(self.chunk_overlap, len(prev_chunk))
                overlap = prev_chunk[-overlap_size:]
                result.append(overlap + chunks[i])
        
        return result 