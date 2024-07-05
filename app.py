import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import UnidentifiedImageError, Image
from model import load_model, load_labels, predict_image

app = Flask(__name__)
CORS(app)

model, device = load_model()
labels = load_labels('labels.txt')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    image_bytes = file.read()
    
    try:
        # Debugging: Check if the image can be opened directly here
        print(f"Read {len(image_bytes)} bytes from the image file.")
        print(f"First 100 bytes of the image: {image_bytes[:100]}")

        image = Image.open(io.BytesIO(image_bytes))
        image_format = image.format
        image_mode = image.mode
        image_size = image.size
        print(f"Image Format: {image_format}, Image Mode: {image_mode}, Image Size: {image_size}")

        predicted_label, confidence_score = predict_image(model, device, image_bytes, labels)
        return jsonify({
            'predicted_label': predicted_label,
            'confidence': confidence_score
        })
    except UnidentifiedImageError:
        return jsonify({'error': 'Cannot identify image file'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'Test endpoint working'})

if __name__ == '__main__':
    app.run(debug=True)
