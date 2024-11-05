from flask import Flask, request, jsonify
import easyocr
import os

app = Flask(__name__)

@app.route('/read_ocr', methods=['POST'])
def read_ocr():
    file = request.files['file']
    file_content = file.read()
    with open('test.jpg', 'wb') as f:
        f.write(file_content)
    reader = easyocr.Reader(['en'])
    result = reader.readtext('test.jpg')
    result = ' '.join([x[1] for x in result])
    os.remove('test.jpg')
    return jsonify({'text': result})
    
if __name__ == '__main__':
    app.run(debug=True)