import easyocr
import os
import uuid

from flask import Blueprint, request, jsonify
from config import config

ocr_bp = Blueprint('ocr', __name__)
reader = easyocr.Reader(['en'], gpu=False)

@ocr_bp.route('/ocr', methods=['POST'])
def read_ocr():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    unique_filename = f"{uuid.uuid4()}.jpg"
    
    try:
        file.save(unique_filename)
        
        result = reader.readtext(unique_filename)
        extracted_text = ' '.join([text for _, text, _ in result])
        
        return jsonify({"text": extracted_text})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    finally:
        if os.path.exists(unique_filename):
            os.remove(unique_filename)