FROM python:3.9-slim

WORKDIR /app

COPY yt_chat.py get_key.py init_db.py /app/
COPY history /app/history
COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

RUN python init_db.py

ENV FLASK_APP=yt_chat.py

EXPOSE 5000 5001

CMD ["sh", "-c", "flask run --host=0.0.0.0 --port=5000 & flask run --host=0.0.0.0 --port=5001 --app get_key.py"]