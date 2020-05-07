import pandas as pd
import numpy as np

import tensorflow as tf

MAX_LEN_TITLE = 82
MAX_LEN_TEXT = 390


def replace_string(df, index, column='Text', replace="", by=""):
    """
    replace specified string from a dataframe entry
    """
    df.loc[index][column] = df.loc[index][column].replace(replace, by)

    return df


def replace_whole(df, replace, by=''):
    """
    Replaces specified string from the entire dataframe
    """
    df.Title = df.Title.str.replace(replace, by, regex=False)
    df.Text = df.Text.str.replace(replace, by, regex=False)

    return df


def clean_data(df):
    """
    Clears the dataframe, removing undesired text from text and title fields
    """
    #  df = replace_string(df, index=, column='Text', replace=)
    df = replace_string(df, index=17438, column='Text',
                        replace=" (pronounced /ˈtæntɪˌvi 'θri:/)")  # Tantive III
    df = replace_string(df, index=13840, column='Text',
                        replace=" (pronounced /'pri vizlɑ/)")  # Pre Visla
    df = replace_string(df, index=13885, column='Text',
                        replace='−', by='-')  # Prime Sabacc
    df = replace_string(df, index=14877, column='Text',
                        replace='°')  # Rippinnium
    df = replace_string(df, index=11779, column='Text',
                        replace=' (pronounced /ˈmoʊˌmɪn/)')  # Momin
    df = replace_string(df, index=8495, column='Text',
                        replace='“', by='"')  # Iloh
    df = replace_string(df, index=8495, column='Text',
                        replace='”', by='"')  # Iloh
    df = replace_string(df, index=1763, column='Text',
                        replace=' (Δ)')  # Base Delta Zero
    df = replace_string(df, index=12597, column='Text',
                        replace='―', by='-')  # Nuhj
    df = replace_string(df, index=3480, column='Text',
                        replace=' (pronounced /kaɪˈmɪərə/)', by='')  # Chimaera
    df = replace_string(df, index=13115, column='Text', replace='Reublic’s clone troopers and the Galactic Empire’s stormtroopers',
                        by="Republic's clone troopers and the Galactic Empire's")  # Ozeer Tenzer
    df = replace_string(df, index=17144, column='Text',
                        replace=', with two "="s in red, one on each side')  # Support The Boys In White
    df = replace_string(df, index=20135, column='Text',
                        replace='=', by='equals')  # Vote Yes on Prop 31-814D
    df = replace_string(df, index=4480, column='Text',
                        replace=' (pronounced /sɪndə/)')  # Cynda
    df = replace_string(df, index=4979, column='Text',
                        replace='%', by=' percent')
    df = replace_string(df, index=6626, column='Text', replace=';',
                        by=',')  # Form VII
    df = replace_string(df, index=13676, column='Text', replace=';',
                        by=',')

    # replaces specified characters from Text and Title columns
    df = replace_whole(df, replace='â', by='a')
    df = replace_whole(df, replace='ó', by='o')
    df = replace_whole(df, replace='ç', by='c')
    df = replace_whole(df, replace='ñ', by='n')
    df = replace_whole(df, replace='æ', by='ae')
    df = replace_whole(df, replace='á', by='a')
    df = replace_whole(df, replace='^', by='')
    df = replace_whole(df, replace='Î', by='I')
    df = replace_whole(df, replace='#', by='')
    df = replace_whole(df, replace='+', by='')
    df = replace_whole(df, replace='è', by='e')
    df = replace_whole(df, replace='&', by='and')
    df = replace_whole(df, replace='é', by='e')
    df = replace_whole(df, replace='–', by='-')
    df = replace_whole(df, replace='—', by='-')
    df = replace_whole(df, replace='  ', by=' ')
    df = replace_whole(df, replace='[source?]')
    df = replace_whole(df, replace=';', by='.')
    
    drop_list = [20378, 1264, 5220, 7109, 8377, 9250, 17081, 20276]
    for index in drop_list:
        df.drop(index, inplace=True)
    
    for i in range(20, 100, 1):
        df = replace_whole(df, replace='[' + str(i) + ']')

    df = replace_whole(df, replace='[')
    df = replace_whole(df, replace=']')

    # adds the 'end of sentence' char to the Title column
    df.Title = df.Title + '\n'

    df.Text = df.Text.str.split('.').map(lambda x: x[0] + '.\n')

    df.reset_index(drop=True, inplace=True)

    df['Title'] = df.Title.str.lower()  # lower all characters
    df['Text'] = df.Text.str.lower()

    return df


def char2int(df, vocab):
    '''
    Takes the dataframe and transforms it to vectors of ints.
    Returns encoder_input_vector, decoder_input_vector, output_vector, each representing 
    '''

    # char to int encoding for the encoder input
    char2int = dict((c, i) for i, c in enumerate(vocab))

    # strings encoded as vectors of integers. They are initialized as arrays of -1, because thats the value for the [NO CHARACTER] char.
    # as we fill these vectors with integers, the -1's will be replaced by the right integers
    encoder_input_vector = np.zeros((df.size, MAX_LEN_TITLE), dtype=np.int8)
    decoder_input_vector = np.zeros((df.size, MAX_LEN_TEXT), dtype=np.int8)
    output_vector = np.zeros((df.size, MAX_LEN_TEXT), dtype=np.int8)

    for example, article in df.iterrows():

        for position, char in enumerate(article.Title):
            encoder_input_vector[example][position] = char2int[char]

        # inserts the [START] character on every decoder input sentence
        decoder_input_vector[example][0] = char2int['[START]']
        for position, char in enumerate(article.Text):
            decoder_input_vector[example][position + 1] = char2int[char]
            output_vector[example][position] = char2int[char]

    # arrays of integers
    return (encoder_input_vector, decoder_input_vector, output_vector)


def dataset_generator(int_encoder_input, int_decoder_input, int_target,
                      size_dict, seed=None):
    '''
    Generator that will be used by tf.data.Dataset to generate the dataset. This uses a character level, one-hot encoding approach
    '''

    n_examples = len(int_encoder_input)

    if seed:
        np.random.seed(seed)

    # randomly shuffles the dataset at each epoch
    indexes = np.random.permutation(n_examples)
    int_encoder_input = int_encoder_input[indexes]
    int_decoder_input = int_decoder_input[indexes]
    int_target = int_target[indexes]

    for example in range(n_examples):  # iterate through all examples

        # input of encoder part of model
        encoder_input = np.zeros([MAX_LEN_TITLE, size_dict])
        # input of decoder part of model
        decoder_input = np.zeros([MAX_LEN_TEXT, size_dict])
        target = np.zeros([MAX_LEN_TEXT, size_dict])   # target of model

        # vectorization of encoder input
        for position, index in enumerate(int_encoder_input[example]):
            encoder_input[position][index] = 1

        # vectorization of decoder input
        for position, index in enumerate(int_decoder_input[example]):
            decoder_input[position][index] = 1

        # vectorization of output
        for position, index in enumerate(int_target[example]):
            target[position][index] = 1

        yield ({'encoder_input': encoder_input, 'decoder_input': decoder_input}, {'output': target})


def build_datasets(df, seed=None, validation_samples=3000, batch=1, vocab=None):
    '''
    Builds Datasets (training and validation datasets)
    seed: seed to make the reordering reproducible
    validation_samples: int that determines how much of the data is going to the validation 
    batch: defines the how many elements each batch in the training dataset will have 
    '''
    
    # randomly reorder dataframe.
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    validation_df = df.iloc[:validation_samples].reset_index(drop=True)
    training_df = df.iloc[validation_samples:].reset_index(drop=True)

    ######### TRAINING DATASET ##########################
    # transforms the dataframe into a array of indexes, each index represeting a char. It has shape of [N_EXAMPLES, SENT_LEN]
    (train_int_encoder_input, train_int_decoder_input,
     train_int_output) = char2int(training_df, vocab=vocab)

    # dataset consists of 2 dicts, inputs = {'encoder_input': [SENT_LEN, VOCAB_SIZE], 'decoder_input': [SENT_LEN, VOCAB_SIZE]} and outputs = {'output': [SENT_LEN, VOCAB_SIZE]}
    # each iteration returns only one triplet
    training_dataset = tf.data.Dataset.from_generator(dataset_generator, output_types=({'encoder_input': tf.float64, 'decoder_input': tf.float64}, {'output': tf.float64}),
                                                      output_shapes=({'encoder_input': tf.TensorShape((None, None)), 'decoder_input': tf.TensorShape((None, None))}, {'output': tf.TensorShape((None, None))}), args=(
        train_int_encoder_input, train_int_decoder_input, train_int_output, len(vocab), seed))

    training_dataset = training_dataset.batch(batch)
    training_dataset = training_dataset.repeat()

    ######### VALIDATION DATASET ##########################
    # transforms the dataframe into a array of indexes, each index represeting a char. It has shape of [N_EXAMPLES, SENT_LEN]
    (val_int_encoder_input, val_int_decoder_input,
     val_int_output) = char2int(validation_df, vocab=vocab)

    # dataset consists of 2 dicts, inputs = {'encoder_input': [SENT_LEN, VOCAB_SIZE], 'decoder_input': [SENT_LEN, VOCAB_SIZE]} and outputs = {'output': [SENT_LEN, VOCAB_SIZE]}
    # each iteration returns only one triplet
    validation_dataset = tf.data.Dataset.from_generator(dataset_generator, output_types=({'encoder_input': tf.float64, 'decoder_input': tf.float64}, {'output': tf.float64}),
                                                        output_shapes=({'encoder_input': tf.TensorShape((None, None)), 'decoder_input': tf.TensorShape((None, None))}, {'output': tf.TensorShape((None, None))}), args=(
        val_int_encoder_input, val_int_decoder_input, val_int_output, len(vocab)))

    validation_dataset = validation_dataset.batch(batch)
    validation_dataset = validation_dataset.repeat()

    return training_dataset, validation_dataset


def one_hot2char(array, vocab):
    '''
    Convert sentences encoded as one_hot vectorization back to strings, given a vocabulary
    array has shape = [N_EXAMPLES, SENTENCE_LENGTH, VOCAB_SIZE]
    Returns a list of strings
    '''

    int2char = dict((int(i), c) for i, c in enumerate(vocab))

    # array of ints, shape = [N_EXAMPLES, SENTENCE_LENGHT]
    ints = np.argmax(array, axis=-1)
    strings = []
    for vector in ints:
        string = ''.join([int2char[index] for index in vector])
        # strings.append(string.split('\n')[0])
        strings.append(string)
    return strings


def process_input(strings, vocab):
    '''
    Converts a string to be used as a input (article title) by the model
    '''
    # char to int encoding for the encoder input
    char2int = dict((c, i) for i, c in enumerate(vocab))

    one_hot = np.zeros([len(strings), MAX_LEN_TITLE, len(vocab)])

    for example, string in enumerate(strings):
        string = string + '\n'
        string = string.ljust(MAX_LEN_TITLE, vocab[0])
        for position, char in enumerate(string):
            if char not in char2int:
                raise Exception('String "{}" contains invalid character "{}".'.format(
                    string.split('\n')[0], char))
            one_hot[example][position][char2int[char]] = 1

    return one_hot
