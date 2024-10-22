FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN wget https://huggingface.co/Mozilla/llava-v1.5-7b-llamafile/resolve/main/llava-v1.5-7b-q4.llamafile
RUN chmod +x llava-v1.5-7b-q4.llamafile
RUN ./llava-v1.5-7b-q4.llamafile

EXPOSE 5000

RUN python init_db.py

CMD ["python", "app.py"]
