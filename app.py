from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import logging

app = Flask(__name__) #instantiating a Flsak application

# Load your trained model
MODEL_PATH = 'C:\\Users\\otien\\potatoDiseas\\training\\potato_d.h5'
model = load_model(MODEL_PATH)

# Config variables
TARGET_SIZE = (256, 256)

# Custom decoding function
def custom_decode_predictions(preds, top=1):
    preds = np.array(preds)
    class_labels = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
    results = []

    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [(class_labels[i], pred[i]) for i in top_indices]
        results.append(result)

    return results

def model_predict(img_path, model):
    try:
        img = image.load_img(img_path, target_size=TARGET_SIZE)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0  # Normalize the pixel values

        # Make prediction
        preds = model.predict(x)

        # Use the custom decoding function
        decoded_preds = custom_decode_predictions(preds, top=1)

        # Extract the class label directly
        result = decoded_preds[0][0][0]

        # Log predictions for debugging
        logging.info(f"Predicted class for {img_path}: {result}")

        return result

    except Exception as e:
        logging.error(f"Error in model prediction: {str(e)}")
        return "Error in model prediction"

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST': 
        try:
            # Get the file from the post request
            f = request.files['file']

            # Save the file to ./uploads
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)

            # Make prediction
            preds = model_predict(file_path, model)

            # Render the result template
            return render_template('result.html', prediction=preds, img_path=file_path)

        except Exception as e:
            logging.error(f"Error in file upload/prediction: {str(e)}")
            return "Error in file upload/prediction"

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
