import re
import keras as k
import numpy as np
import gensim

from nltk.stem.isri import ISRIStemmer

from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense, Dropout, GRU
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


#written by Ahmed Emad May 2018
np.random.seed(52) #for reproductaibility


# Clean/Normalize Arabic Text
def clean_str(text):
    # written by Abo Bakr ... a little modification by Ahmed

    search = ["أ", "إ", "آ", "ة", "_", "-", "/", ".", "،", " و ", " يا ", '"', "ـ", "'", "ى", "\\", '\n', '\t', '&quot;', '?', '؟', '!', '«', '»']
    replace = ["ا", "ا", "ا", "ه", " ", " ", " ", " ", " ", " و", " يا", "", "", "", "ي", "", ' ', ' ', ' ', ' ? ', ' ؟ ', ' ! ', ' ', ' ']

    # remove tashkeel
    p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(p_tashkeel, "", text)

    text = text.replace('وو', 'و')
    text = text.replace('يي', 'ي')
    text = text.replace('اا', 'ا')

    for i in range(0, len(search)):
        text = text.replace(search[i], replace[i])

    # remove urls
    p_url = re.compile(r'([A-Za-z]+://)?[A-Za-z0-9-_]+.[A-Za-z0-9-_:%&~?/.=]+')
    text = re.sub(p_url, ' ', text)

    # remove user mentions
    p_mention = re.compile(r'[@]+[\u0627-\u064aA-Za-z0-9_:]+')
    text = re.sub(p_mention, ' ', text)

    # remove latin
    latin_pattern = re.compile(r'[a-zA-Z]+')
    text = re.sub(latin_pattern, '', text)

    # remove longation
    p_longation = re.compile(r'(.)\1+')
    subst = r"\1\1"
    text = re.sub(p_longation, subst, text)

    # trim
    text = text.strip()

    return text


def read_data_from_file(filename): #"reading and preparing the training or testing datasets by extracting and separating tweets from labels."

    tweets = []  # list of text samples
    labels = []  # list of label ids
    labels_index = {}  # dictionary mapping label name to numeric id


    istemmer = ISRIStemmer()
    read_file = open(filename, "r+")  # read and write mode

    index = 0
    for line in read_file:

        line = line.split('\t')  # to get the tweet itself

        label = line[0]
        tweet = line[1].strip(" \"")

        tweet = clean_str(tweet)
        tweet = istemmer.norm(tweet)

        if (label not in labels_index):
            labels_index[label] = index
            index += 1

        tweets.append(tweet)
        labels.append(labels_index[label])

    read_file.close()

    return [tweets, labels]


#training
def fit(model, training_set, training_label, BATCH_SIZE, testing_set, testing_label):

    Number_Batches = int(len(training_set) / BATCH_SIZE)
    records_covered = Number_Batches * BATCH_SIZE
    extra_batch = 0  # in case records didn't fit all in batches

    # Variable-length input: 'fit' and 'evaluate' in Keras take 2D arrays of the same length
    for epoch in np.arange(EPOCHS):

        score = np.asarray([0.0, 0.0])

        for i in np.arange(1, Number_Batches + 1):
            batch_sequences = np.asarray(training_set[(i - 1) * BATCH_SIZE: i * BATCH_SIZE])
            batch_sequences = pad_sequences(batch_sequences)  #Pad to maximum length per each batch

            batch_labels = training_label[(i - 1) * BATCH_SIZE: i * BATCH_SIZE]

            model.train_on_batch(batch_sequences, batch_labels)
            score += evaluate_on_batch(model, batch_sequences, batch_labels)

        if (len(training_set) > records_covered):  # for remaining records
            batch_sequences = np.asarray(training_set[records_covered: len(training_set)])
            batch_sequences = pad_sequences(batch_sequences)  # Pad to maximum length per each batch

            batch_labels = training_label[records_covered: len(training_set)]

            model.train_on_batch(batch_sequences, batch_labels)
            score += evaluate_on_batch(model, batch_sequences, batch_labels)

            extra_batch = 1

        score[0] /= (Number_Batches + extra_batch)
        score[1] /= (Number_Batches + extra_batch)
        print("Epoch: %s/%s ... Loss: %.2f ... Acc: %.3f" % ((epoch + 1), EPOCHS, score[0], score[1]))

    evaluate(model, testing_set, testing_label)


#evaluate
def evaluate_on_batch(model, x_set, y_set):

    y_pred = model.predict(x_set)

    predictions = np.argmax(y_pred, axis=1)  # convert to classes
    y_set = np.argmax(y_set, axis=1)

    acc = np.mean(predictions == y_set)
    losses = y_pred[np.arange(len(y_set)), y_set]
    loss = (-1 * np.sum(np.log(losses))) / len(predictions)

    return [loss, acc]

#evaluate
def evaluate(model, x_set, y_set):

    y_pred = np.empty(y_set.shape)

    for i in np.arange(0, len(x_set)):
        y_pred[i] = model.predict(np.asarray(x_set[i]).reshape(1, len(x_set[i])))

    predictions = np.argmax(y_pred, axis=1)  # convert to classes
    y_set = np.argmax(y_set, axis=1)

    acc = np.mean(predictions == y_set)
    losses = y_pred[np.arange(len(y_set)), y_set]
    loss = (-1 * np.sum(np.log(losses))) / len(predictions)

    print("Testing Acc: %.3f .. Testing loss: %s " %(acc, loss,))

#######################################################

BATCH_SIZE = 64
EMBEDDING_VECTOR_SIZE = 300
HIDDEN_SIZE = 128
EPOCHS = 10
NUMBER_CLASSES = 10 #topics

TRAINING_SAMPLES = 5626
TESTING_SAMPLES = 1000

print("loading and processing text dataset ....")

dataset, labels = read_data_from_file("Datasets/topic_dataset.txt")

print("Converting to sequence & building embedding matrix...")
# finally, vectorize the text samples into a 2D integer tensor

tokenizer = Tokenizer(filters='"#$%&()*+,-./:;<=>@[\]^_`{|}~ ', oov_token='UNK')
tokenizer.fit_on_texts(dataset)
sequences = tokenizer.texts_to_sequences(dataset)

word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))

#prepare embedding matrix
emb_model = gensim.models.Word2Vec.load('/mydata/tweets_sg_300/tweets_sg_300')
embedding_matrix = np.zeros((len(word_index)+1, EMBEDDING_VECTOR_SIZE))
OOTV_perc = 0

for word, i in word_index.items():
    try:
        embedding_vector = emb_model.wv[word]
        embedding_matrix[i] = embedding_vector
        OOTV_perc += 1
    except KeyError:
        embedding_matrix[i] = embedding_matrix[1] #UNK
        pass


print('OOTV Percentage. %s #' % int((1-OOTV_perc/len(word_index)) * 100))


print("Vectoring training data and preparing batches...")

sequences = np.asarray(sequences)
labels = np.asarray(labels)
labels = k.utils.to_categorical(labels, num_classes= NUMBER_CLASSES)

indices = np.arange(len(sequences))
np.random.shuffle(indices)

sequences = sequences[indices] #shuffle data
labels = labels[indices]

training_set = sequences[:TRAINING_SAMPLES]
training_label = labels[:TRAINING_SAMPLES]

testing_set = sequences[-TESTING_SAMPLES:]
testing_label = labels[-TESTING_SAMPLES:]


print('Training model...')

model = Sequential()
model.add(Embedding(len(word_index)+1,
                    EMBEDDING_VECTOR_SIZE,
                    weights=[embedding_matrix],
                    trainable=False))

model.add(GRU(HIDDEN_SIZE, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(NUMBER_CLASSES, activation='softmax')) #8 topics

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#training
fit(model, training_set, training_label, BATCH_SIZE, testing_set, testing_label) #should be implemented using callbacks

#testing
evaluate(model, testing_set, testing_label)


print('Saving model...')

#serialize model to json
model_json = model.to_json()
with open ("/output/topicModel.json","w") as json_file:
    json_file.write(model_json)

#serialize weights to HDf5
model.save_weights("/output/topicModel.h5")
print("Saved model to disk")

'''
loading and processing text dataset ....
Converting to sequence & building embedding matrix...
Found 24587 unique tokens.
OOTV Percentage. 20 #
Vectoring training data and preparing batches...
Training model...
Testing Acc: 0.825 .. Testing loss: 0.5986284236345842
Saving model...
Saved model to disk

'''


