from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello():
    return jsonify(message='Hello, Ollama!')


@app.route('/api/search', methods=['GET'])
def genAI():
    input_text = request.args.get('text', 'no data')
    print(input_text)
    # Process the input_text (you can replace this with your logic)
    url = "http://ollama:11434/api/generate"

    # Sample JSON payload
    payload = {"model": "llama2:7b", "prompt": input_text, "stream": False, "format": "json"}

    # Make a POST request with JSON payload
    response = requests.post(url, json=payload)
    
    # Return a JSON response
    return jsonify({'status': response.status_code, "message": response.json()})

if __name__ == '__main__':
    app.run(debug=True, host ='0.0.0.0', port=6000)
