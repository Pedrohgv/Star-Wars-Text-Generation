
from flask import Flask, request, render_template, make_response, session

import tensorflow as tf
from tensorflow.keras.models import load_model
from model_loader import generate_text


app = Flask(__name__, static_url_path='/static')
app.secret_key = 'key'

model_folder = 'deploy/model'
model = load_model(model_folder + '/trained_model.h5')  # loads the model


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        seed_string = request.form['seedString']
        seed_string = seed_string.lower()
        generated_text = generate_text(seed_string)
        print(generated_text)
        generated_text = generated_text.split('\n')[0]
        return render_template('template.html', generated_text=generated_text)
    else:
        return render_template('template.html')


print(app.static_folder)
if __name__ == '__main__':
    app.run(debug=True)
