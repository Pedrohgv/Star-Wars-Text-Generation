# Star-Wars-Text-Generation
## Summary

This project is an experiment on LSTM's and how to train a language model to generate texts within a specific domain at a character level. Given a seed title, it writes a brief description of it through an API. The model was built using **Tensorflow** from scratch, without transfer learning. It uses texts from Wookiepedia.com.

## Motivation
I am, without doubt, one of those human beings known as "Star Wars fans". I remember watching *The Phantom Menace* on TV and being blown away by it when I was a kid (I know, it is not a great movie). Later, I watched *Attack of the Clones* and *Revenge of the Sith* (this last on the theater) and went on to watch the Original Trilogy on DVD, back when those were still around. This culminated with me, as an adult, having a tattoo of a Tie Fighter on my right arm.


This passion for Star Wars was a good answer to the question "what NLP project should I do?"; I wanted to develop a complete Natural Language Processing project, practicing several skills while doing it.

## Final Product
A demonstration of the working model can be seen bellow.

<img src="images/demonstration.gif" alt="drawing" width="900" height="444"/>

## Requirements
In order to replicate this model, you need to download the code from [here](https://github.com/Pedrohgv/Star-Wars-Text-Generation). Then, install all needed dependencies (it is very recommended to create a new virtual environment and install all packages on it) if you are using Linux with: 
    
    pip install -r requirements-linux.txt

If you are on Windows however, just use:

    pip install -r requirements-windows.txt 

After all dependencies are installed, you need the trained model to be able to generate any text. Because of size restrictions on GitHub, the model must be downloaded from [here](https://drive.google.com/drive/folders/1JTzVV8uir74BlqF9HXtYMZdIu_4ZDpBP?usp=sharing). Just download the entire *model* folder and put it under the *deploy* folder. Finally, with the environment you installed all dependencies on activated and with an open terminal on the folder you downloaded the project, type:

    python deploy/deploy.py



## The Project

Bellow is a list of all the important libraries used:

- BeautifulSoup
- Requests
- Lxml
- Pandas
- Numpy
- MatplotLib
- Tensorflow
- Flask
- HTML
- CSS


We will now go through all the main parts of the project.

### Data Handling

The texts used to train the model were mined from the [Wookiepedia website](https://starwars.fandom.com/wiki/Main_Page) (a sort of Star Wars Wikipedia) using web scrapping.. All the code used for this task is on the [wookiescraper.py](https://github.com/Pedrohgv/Star-Wars-Text-Generation/blob/master/wookiescraper.py) file and a class `Article` was created to structure the texts, with each article containing a title, a description of the subject (this will be the first and brief description on the article's page), and the categories in which they belong. The main libraries for extracting the data were **beautifulsoup**, **requests** and **Pandas** (to store texts in a dataframe).
 
In order to list all possible articles, the function `create_complete_database` is called. It creates a list of URL's of all *Canon* articles (in the Star Wars Universe there was a reboot of all stories that were produced in alternative media like books, comics and video games; the old stories were labeled as "*Legends*" while the stories that remained official and new stories are considered to be *Canon*). Then, it downloads and creates each article on the list by using the *Article* class own functions. A dataframe is then created containing all downloaded articles and saved on a *Complete Database.csv* file.

To feed the model, we must also process the acquired data; the file [data_processor.py](https://github.com/Pedrohgv/Star-Wars-Text-Generation/blob/master/data_processor.py) contains all the code used for this task. The function `clean_data` takes a dataframe of texts and formats it by removing undesired characters and shortening texts (this must be done because since we are creating a model that works on a character level, the model will have a hard time learning patterns and context from longer sentences). The file also contains functions that transforms a given text corpus into *one hot* vectors and vice-versa, as well as *dataset generator* functions; instead of loading all data into memory during training, the function `build_datasets` will build a *training_dataset* and a *validation_dataset*, each one being a Tensorflow *Dataset* object that process and feeds data into the model as chunks during training.

### Model and Training

[LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) is a type of Recurrent Neural Network cell that has a higher capacity of holding information over long sequences of data. Because of this feature, it is very useful when dealing with NLP problems. For this project, a [sequence-to-sequence](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html) architecture was used to generate the output sentences. In this approach, a input string (article's title) is given to an encoder that processes the data sequentially character by character, and delivers an encoded vector that contains information about the input. Then, a decoder will use this information to generate, again sequentially and character by character, a new embedded vector that will then go into a **softmax** layer to produce a vector of probabilities, one value for each possible character. Each output character is generated after the previous one, always using also the vector containing the input information generated by the encoder. The code for creating and training the model is in the file [run.py](https://github.com/Pedrohgv/Star-Wars-Text-Generation/blob/master/run.py).

After GPU initialization and vocabulary definition, a `config` dictionary was created in order to make hyperparameter tunning easier:

```python
# enable memory growth to be able to work with GPU
GPU = tf.config.experimental.get_visible_devices('GPU')[0]
tf.config.experimental.set_memory_growth(GPU, enable=True)

# set tensorflow to work with float64
tf.keras.backend.set_floatx('float64')

# the new line character (\n) is the 'end of sentence', therefore there is no need to add a '[STOP]' character
vocab = 'c-y5i8"j\'fk,theqm:/.wnlrdg0u1 v\n4b97)o36z2axs(p'
vocab = list(vocab) + ['[START]']

config = {  # dictionary that contains the training set up. Will be saved as a JSON file
    'DIM_VOCAB': len(vocab),
    'MAX_LEN_TITLE': MAX_LEN_TITLE,
    'MAX_LEN_TEXT': MAX_LEN_TEXT,
    'DIM_LSTM_LAYER': 512,
    'ENCODER_DEPTH': 2,
    'DECODER_DEPTH': 2,
    'LEARNING_RATE': 0.0005,
    'BATCH_SIZE': 16,
    'EPOCHS': 100,
    'SEED': 1,
    # 'GRAD_VAL_CLIP': 0.5,
    # 'GRAD_NORM_CLIP': 1,
    'DECAY_AT_10_EPOCHS': 0.9,
    'DROPOUT': 0.2,
}
```

A seed will also be used in order to make results reproducible:

```python
tf.random.set_seed(config['SEED'])
```

After that, finally, the acquired data is loaded into a dataframe, cleaned, and new configuration options can be set now, like the train/validation split:

```python
data = pd.read_csv('Complete Database.csv', index_col=0)

data = clean_data(data)

config['STEPS_PER_EPOCH'] = int((data.shape[0] - 1500) / config['BATCH_SIZE'])
config['VALIDATION_SAMPLES'] = int(
    data.shape[0]) - (config['STEPS_PER_EPOCH'] * config['BATCH_SIZE'])
config['VALIDATION_STEPS'] = int(
    np.floor(config['VALIDATION_SAMPLES'] / config['BATCH_SIZE']))
```

The learning rate is then defined. In this project, an **exponentially decaying** learning rate was shown to give best results:

```python
# configures the learning rate to be decayed by the value specified at config['DECAY_AT_10_EPOCHS'] at each 10 epochs, but to that gradually at each epoch
learning_rate = ExponentialDecay(initial_learning_rate=config['LEARNING_RATE'],
                                decay_steps=config['STEPS_PER_EPOCH'],
                                decay_rate=np.power(
                                    config['DECAY_AT_10_EPOCHS'], 1/10),
                                staircase=True)
```

With the data loaded, both training and validation datasets can now be built:

```python
training_dataset, validation_dataset = build_datasets(
    data, seed=config['SEED'], validation_samples=config['VALIDATION_SAMPLES'], batch=config['BATCH_SIZE'], vocab=vocab)
```

With the goal of saving the model after training, a path will be chosen. This folder will be named with a generic name, and then changed to the specific time the model finished training. Also, both vocabulary and model configuration will be saved as `json` files.

```python
folder_path = 'Training Logs/Training'  # creates folder to save traning logs
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# saves the training configuration as a JSON file
with open(folder_path + '/config.json', 'w') as json_file:
    json.dump(config, json_file, indent=4)
    
# saves the vocab used as a JSON file
with open(folder_path + '/vocab.json', 'w') as json_file:  
    json.dump(vocab, json_file, indent=4)
```

For monitoring the model, some Callback functions (functions that are called during training, at specified intervals) were used. These functions are contained in the [callbacks.py](https://github.com/Pedrohgv/Star-Wars-Text-Generation/blob/master/callbacks.py) file. A custom class `CallbackPlot` was created, in order to plot the training error throughout training. Objects of Tensorflow Callback classes `ModelCheckpoint` and `CSVLogger` were also instantiated, in order to save the model and training logs during training respectively:

```python
loss_plot_settings = {'variables': {'loss': 'Training loss',
                                    'val_loss': 'Validation loss'},
                    'title': 'Losses',
                    'ylabel': 'Epoch Loss'}

last_5_plot_settings = {'variables': {'loss': 'Training loss',
                                    'val_loss': 'Validation loss'},
                        'title': 'Losses',
                        'ylabel': 'Epoch Loss',
                        'last_epochs': 5}

plot_callback = CallbackPlot(folder_path=folder_path,
                            plots_settings=[
                                loss_plot_settings, last_5_plot_settings],
                            title='Losses', share_x=False)

model_checkpoint_callback = ModelCheckpoint(
    filepath=folder_path + '/trained_model.h5')

csv_logger = CSVLogger(filename=folder_path +
                    '/Training logs.csv', separator=',', append=False)
```

Finally, the model can be built, compiled, and trained:

```python
###### BUILDS MODEL FROM SCRATCH WITH MULTI LAYER LSTM ###########
tf.keras.backend.clear_session()  # destroys the current graph

encoder_inputs = Input(shape=(None, config['DIM_VOCAB']), name='encoder_input')
enc_internal_tensor = encoder_inputs

for i in range(config['ENCODER_DEPTH']):
    encoder_LSTM = LSTM(units=config['DIM_LSTM_LAYER'],
                        batch_input_shape=(
                            config['BATCH_SIZE'], MAX_LEN_TITLE, enc_internal_tensor.shape[-1]),
                        return_sequences=True, return_state=True,
                        name='encoder_LSTM_' + str(i), dropout=config['DROPOUT'])
    enc_internal_tensor, enc_memory_state, enc_carry_state = encoder_LSTM(
        enc_internal_tensor)  # only the last states are of interest

decoder_inputs = Input(shape=(None, config['DIM_VOCAB']), name='decoder_input')
dec_internal_tensor = decoder_inputs

for i in range(config['DECODER_DEPTH']):
    decoder_LSTM = LSTM(units=config['DIM_LSTM_LAYER'],
                        batch_input_shape=(
                            config['BATCH_SIZE'], MAX_LEN_TEXT, dec_internal_tensor.shape[-1]),
                        return_sequences=True, return_state=True,
                        name='decoder_LSTM_' + str(i), dropout=config['DROPOUT'])  # return_state must be set in order to retrieve the internal states in inference model later
    # every LSTM layer in the decoder model have their states initialized with states from last time step from last LSTM layer in the encoder
    dec_internal_tensor, _, _ = decoder_LSTM(dec_internal_tensor, initial_state=[
                                            enc_memory_state, enc_carry_state])

decoder_output = dec_internal_tensor

dense = Dense(units=config['DIM_VOCAB'], activation='softmax', name='output')
dense_output = dense(decoder_output)

model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=dense_output)

model.compile(optimizer=optimizer, loss='categorical_crossentropy')

history = model.fit(x=training_dataset,
                    epochs=config['EPOCHS'],
                    steps_per_epoch=config['STEPS_PER_EPOCH'],
                    callbacks=[plot_callback, csv_logger,
                            model_checkpoint_callback],
                    validation_data=validation_dataset,
                    validation_steps=config['VALIDATION_STEPS'])
```
After training, a folder will contain all data relevant to this training session, like loss function plots, error at different time steps, and the model itself.

```python
model.save(folder_path + '/trained_model.h5', save_format='h5')
model.save_weights(folder_path + '/trained_model_weights.h5')
plot_model(model, to_file=folder_path + '/model_layout.png', show_shapes=True, show_layer_names=True, rankdir='LR')


timestamp_end = datetime.now().strftime('%d-%b-%y -- %H:%M:%S')

# renames the training folder with the end-of-training timestamp
root, _ = os.path.split(folder_path)

timestamp_end = timestamp_end.replace(':', '-')
os.rename(folder_path, root + '/' + 'Training Session - ' + timestamp_end)

print("Training Successfully finished.")
```

### Deployment

In order to serve the model, a simple interface was built using the **Flask** package, and it's code can be found under the [deploy](https://github.com/Pedrohgv/Star-Wars-Text-Generation/tree/master/deploy) folder.

## Conclusion

After training, the model is able to generate sentences given a seed string. Bellow are some examples with the produced sentence and the seed string used to generate it (I would use the names of my cats as examples, but since they are already called Luke, Han, and Leia, they're already present in the training dataset):

- ***pedro***: pedra was a human male who served as a commanarr in the reb l alanet oernie.
- ***pedro henrique***: pedra oesoinon was a human female who served as a commtnan the gabal formls of the galactic republic.
- ***chico***: chics was a tama ium ted who served the galactic empire as a conmanan ic the galactic empire.
- ***sara***: sara was a human male who served as a commandrr in the rew lepublic during the galactic civil war.

As can be seen above, the model learned how to form some words reasonably well, how to size those words, how to end a sentence properly and how to form some sort of context. However, it seems heavily biased towards always describing humans serving under a faction within the Star Wars Universe. This can be explained by the fact that the model's architecture wasn't build using word embeddings (which would allow for more complex context learning) because there are several words that are unique to Star Wars. I nice idea for a future project would be to generate a specific word embedding for the Star Wars Universe, and then use that to generate new text.