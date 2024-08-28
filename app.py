import os
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from keras.models import load_model
from werkzeug.utils import secure_filename  # Import secure_filename here

app = Flask(__name__)
model = load_model('Braintumor10EpochsCategorical.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def get_className(classNo):
    return "Normal" if classNo == 0 else "Brain Tumor"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Load and preprocess image
        image = cv2.imread(file_path)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((64, 64))
        image = np.array(image)
        input_img = np.expand_dims(image, axis=0)
        
        # Predict
        result = model.predict(input_img)
        class_no = np.argmax(result)
        class_name = get_className(class_no)
        
        return render_template('result.html', result=class_name, filename=filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
