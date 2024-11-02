from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('model/trained_plant_disease_model.keras')

# Update the class names based on your new list
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 
    'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

def preprocess_image(image_bytes):
    """Preprocess the uploaded image for model prediction"""
    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    
    # Resize image to match model's expected input size
    image = image.resize((128, 128))  # Adjust the size based on your model's input
    img_array = img_to_array(image)
    img_array = np.array([img_array])  # Convert single image to a batch.
    
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        image_file = request.files['image']
        img_bytes = image_file.read()
        processed_image = preprocess_image(img_bytes)
        
        # Make prediction
        predictions = model.predict(processed_image)
        result_index = np.argmax(predictions[0])
        predicted_class = class_names[result_index]
        confidence = float(np.max(predictions[0]))
        
        return jsonify({
            'status': 'success',
            'prediction': predicted_class,
            'confidence': confidence,
            'recommendations': get_recommendations(predicted_class)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_recommendations(disease_class):
    """Return treatment recommendations based on the detected disease"""
    recommendations = {
        'Apple___Apple_scab': [
            "Remove infected leaves",
            "Apply fungicide",
            "Improve air circulation"
        ],
        'Apple___Black_rot': [
            "Remove affected fruits",
            "Apply copper fungicides",
            "Practice crop rotation"
        ],
        'Apple___Cedar_apple_rust': [
            "Prune cedar trees near apple trees",
            "Apply fungicide during bloom",
            "Improve air circulation"
        ],
        'Apple___healthy': [
            "Continue regular maintenance",
            "Ensure proper watering",
            "Regular pruning"
        ],
        # Add more recommendations for other classes as needed
    }
    return recommendations.get(disease_class, ["Please consult a plant specialist"])

if __name__ == '__main__':
    app.run(debug=True)
