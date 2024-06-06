from generator import tokenize, generate_images
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)


@app.route('/prompt/generate', methods=['POST'])
def prompt_generate():
    try:
        body = request.json

        if body is None:
            return jsonify({'error': 'Invalid request body'}), 400

        prompt = body['prompt']
        count = body['count'] if isinstance(body['count'], int) else 8

        if not prompt or not isinstance(prompt, str):
            return jsonify({'error': 'Prompt must be a non-empty string'}), 400

        if count < 2 or count > 12:
            return jsonify({'error': 'Count must be an integer between 2 and 12'}), 400

        # tokenize prompt
        labels = tokenize(prompt, count)

        if len(labels) == 0:
            return jsonify({'error': 'Invalid prompt.'}), 400

        images = generate_images(labels)

        return jsonify({'image_urls': [f"https://imagine.automos.net/generated/{image}" for image in images]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/label/generate', methods=['POST'])
def label_generate():
    try:
        body = request.json

        if body is None:
            return jsonify({'error': 'Invalid request body'}), 400

        labels = body['labels']

        if len(labels) == 0 or len(labels) > 12:
            return jsonify({'error': 'Invalid labels.'}), 400

        if len([label for label in labels if label < 0 or label > 500]) > 0:
            return jsonify({'error': 'Invalid label.'}), 400

        images = generate_images(labels)

        return jsonify({'image_urls': [f"https://imagine.automos.net/generated/{image}" for image in images]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/generated/<path:filename>')
def serve_image(filename):
    return send_from_directory('generated', filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
