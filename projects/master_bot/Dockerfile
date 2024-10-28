FROM python:3.9-slim

WORKDIR /app

COPY genaitor.py /app/
COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

RUN wget https://huggingface.co/Mozilla/llava-v1.5-7b-llamafile/resolve/main/llava-v1.5-7b-q4.llamafile
RUN chmod +x llava-v1.5-7b-q4.llamafile
RUN ./llava-v1.5-7b-q4.llamafile

RUN python genaitor.py

ENV FLASK_APP=genaitor.py

EXPOSE 5000

CMD ["sh", "-c", "flask run --host=0.0.0.0 --port=5000 --app genaitor.py"]