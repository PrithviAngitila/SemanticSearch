FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# RUN llama2-llm ollama run llama2:7b

COPY . /app
EXPOSE 6000

CMD ["python", "app.py"]
