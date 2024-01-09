import logging
import os
import requests
from flask import Flask, jsonify, request
from pydantic import ValidationError
from validator import Search
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.route("/", methods=["GET"])
def hello():
    return jsonify(message="Hello, Ollama!")


def build_request(query):
    container_name = os.environ.get("CONTAINER_NAME", "")
    port = os.environ.get("LLM_PORT", "")
    url = f"http://{container_name}:{port}/api/generate"
    payload = {
        "model": os.environ.get("MODEL", "llama2"),
        "prompt": query,
        "stream": False,
        "format": "json",
    }
    return url, payload


@app.route("/api/search", methods=["GET"])
def genAI():
    try:
        search = Search.from_dict(request.args)
        # build request
        url, payload = build_request(search.query)
        # Make a POST request with JSON payload
        response = requests.post(url, json=payload)
        status_code = response.status_code
        output = response.json()

        if status_code == 200:
            return jsonify({"status": status_code, "message": output["response"]})
        else:
            return jsonify({"status": status_code, "message": "Internal server error"})

    except ValidationError as e:
        # Log the validation error
        logger.error(f"Validation error: {str(e)}")
        return jsonify({"error": str(e), "status": 400})
    except Exception as ex:
        # Log other exceptions
        logger.exception(f"An error occurred: {str(ex)}")
        return jsonify({"error": "Internal server error", "status": 500})


if __name__ == "__main__":
    app.run(
        debug=os.environ.get("LOG_LEVEL", 'info'),
        host="0.0.0.0",
        port=int(os.environ.get("FLASK_PORT", 5000)),
    )
