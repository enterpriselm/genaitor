from typing import List
import os
from .audio import transcribe_audio
from .video import transcribe_video
from .image import image_to_text
from ..file_processors import (
    extract_text_from_doc,
    extract_text_from_json,
    extract_text_from_pdf,
    extract_text_from_ppt,
    read_csv,
    read_excel
)
from langchain_community.document_loaders import YoutubeLoader

def process_media_files(media_files: List[str]) -> str:
    """
    Process a list of media files and return their combined text content.
    
    Args:
        media_files (List[str]): List of file paths or URLs to process
        
    Returns:
        str: Combined text content from all media files
    """
    text = []
    
    for media in media_files:
        try:
            if media.startswith('https://') and 'youtube.com' in media:
                video_id = media.split('watch?v=')[1]
                loader = YoutubeLoader(video_id)
                text.append(loader.load()[0].page_content)
                
            elif media.endswith(('.mp3', '.wav')):
                text.append(transcribe_audio(media))
                
            elif media.endswith('.mp4'):
                text.append(transcribe_video(media))
                
            elif media.endswith(('.doc', '.docx')):
                text.append(extract_text_from_doc(media))
                
            elif media.endswith('.json'):
                text.append(extract_text_from_json(media))
                
            elif media.endswith('.pdf'):
                text.append(extract_text_from_pdf(media))
                
            elif media.endswith(('.ppt', '.pptx')):
                text.append(extract_text_from_ppt(media))
                
            elif media.endswith(('.xlsx', '.xls')):
                text.append(read_excel(media))
                
            elif media.endswith('.csv'):
                text.append(read_csv(media))
                
            elif media.endswith(('.jpg', '.jpeg', '.png')):
                text.append(image_to_text(media))
                
        except Exception as e:
            print(f"Error processing {media}: {str(e)}")
            continue
            
    return "\n\n".join(text) 