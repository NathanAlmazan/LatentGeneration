from generator import tokenize, generate_images
from flask import Flask, request, jsonify, send_from_directory


app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    body = request.json
    
    if body is None:
        return jsonify({'error': 'Invalid request body'}), 400
    
    prompt = body['prompt']
    
    if not prompt or not isinstance(prompt, str):
        return jsonify({'error': 'Prompt must be a non-empty string'}), 400
    
    # tokenize prompt
    labels = tokenize(prompt)
    images = generate_images(labels)
    
    return jsonify({'image_urls': [ f"https://imagine.automos.net/generated/{image}" for image in images ]})


@app.route('/generated/<path:filename>')
def serve_image(filename):
    return send_from_directory('generated', filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)