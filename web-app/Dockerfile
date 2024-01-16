FROM python:3.12-slim


WORKDIR /app 

RUN apt-get update && \
    apt-get install -y build-essential


COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# RUN llama2-llm ollama run llama2:7b

COPY . /app
EXPOSE 6000

CMD ["gunicorn", "-b", "0.0.0.0:6000", "app:app"]
