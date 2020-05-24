
from flask import Flask, request, render_template, make_response
import tensorflow as tf
from tensorflow.keras.models import load_model
from model_loader import generate_text


# enable memory growth to be able to work with GPU
GPU = tf.config.experimental.get_visible_devices('GPU')[0]
tf.config.experimental.set_memory_growth(GPU, enable=True)


app = Flask(__name__, static_url_path='/static')

model_folder = 'model'
# model = load_model(model_folder + '/trained_model.h5')  # loads the model


@app.route('/', methods=['GET', 'POST'])
def display_gui():
    if request.method == 'POST':
        seed_string = request.form['seedString']
        seed_string = seed_string.lower()
        generated_text = generate_text(seed_string)
        generated_text = generated_text.split('\n')[0]
        return render_template('template.html', generated_text=generated_text)

    return render_template('template.html')


if __name__ == '__main__':
    app.run()
