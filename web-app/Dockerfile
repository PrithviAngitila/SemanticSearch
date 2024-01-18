FROM python:3.11-slim


WORKDIR /app 

RUN apt-get update && \
    apt-get install -y build-essential
# RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# RUN pip install torch torchvision torchaudio

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# RUN llama2-llm ollama run llama2:7b

COPY . /app
EXPOSE 6000

CMD ["gunicorn", "-b", "0.0.0.0:6000", "app:app"]
