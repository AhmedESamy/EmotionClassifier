"""written by Ahmed Emad May 2019"""

import re
import gensim
import  numpy as np
import keras as k

from nltk.stem.isri import ISRIStemmer

from keras.models import Model, Sequential, model_from_json
from keras.layers import Embedding, Dense, Activation, Dropout, GRU, Input,Conv2D, GlobalMaxPool2D, Bidirectional, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import f1_score, jaccard_similarity_score


# Setting the seed for numpy-generated random numbers
np.random.seed(72) #for reproductaibility


# Clean/Normalize Arabic Text
def clean_str(text):
    # written by Abo Bakr ... a little modification by Ahmed

    search = ["أ", "إ", "آ", "ة", "_", "-", "/", ".", "،", " و ", " يا ", '"', "ـ", "'", "ى", "\\", '\n', '\t', '&quot;', '?', '؟', '!', '«', '»']
    replace = ["ا", "ا", "ا", "ه", " ", " ", "", "", "", " و", " يا", "", "", "", "ي", "", ' ', ' ', ' ', ' ? ', ' ؟ ', ' ! ', ' ', ' ']

    # remove tashkeel
    p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(p_tashkeel, "", text)

    # remove longation
    p_longation = re.compile(r'(.)\1+')
    subst = r"\1\1"
    text = re.sub(p_longation, subst, text)

    text = text.replace('وو', 'و')
    text = text.replace('يي', 'ي')
    text = text.replace('اا', 'ا')

    for i in range(0, len(search)):
        text = text.replace(search[i], replace[i])

    # trim
    text = text.strip()

    return text



def read_data_from_file(filename, number_of_classes): #"reading and preparing the training or testing datasets by extracting and separating tweets from labels."

    tweets = []  # list of text samples
    labels = []  # list of label ids

    istemmer = ISRIStemmer()
    read_file = open(filename, "r+")  # read and write mode

    for line in read_file:
        tweet = ""

        filtered_line = line.split()  # to get the tweet itself

        label = list(map(int, filtered_line[-number_of_classes:]))

        for word in filtered_line[1:-number_of_classes]:
            tweet += word + " "

        tweet = tweet[:-1]
        tweet = clean_str(tweet)

        tweet = istemmer.norm(tweet)
        '''
        if (label.__contains__(1)):  # for neutral cases
            label.append(0)
        else:
            label.append(1)
        '''
        tweets.append(tweet)
        labels.append(label)

    read_file.close()

    return [tweets, labels]


def write_to_file (y_pre):
    read_file = open("Datasets/testing_set_Ar.txt", "r+")  # read and write mode
    write_test_file = open("E-C_ar_pred.txt", "w")  # write mode

    i = 0

    for line in read_file:
        tweet = ""

        filtered_line = line.split('\t')  # to get the tweet itself

        tweet = filtered_line[0] + '\t' + filtered_line[1] + '\t'
        tweet += np.array2string(y_pre[i], separator='\t').strip('[]')
        tweet += '\n'
        write_test_file.write(tweet)

        i += 1

    read_file.close()
    write_test_file.close()


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
        if ((epoch+1) in [12, 15, 17, 20, 24]):
            evaluate(model, testing_set, testing_label)


#evaluate
def evaluate_on_batch(model, x_set, y_set):

    y_pred = model.predict(x_set)

    predictions = np.zeros(y_pred.shape)
    predictions[y_pred >= 0.5] = 1  # convert to zeros and ones

    acc = jaccard_similarity_score(y_set, predictions)
    loss = -y_set * np.log(y_pred ) - (1 - y_set) * np.log(1 - y_pred + 10 ** -10)
    total_loss = np.sum(loss)/(y_pred.shape[0]*y_pred.shape[1])

    return [total_loss, acc]

#evaluate
def evaluate(model, x_set, y_set):

    y_pred = np.empty(y_set.shape)

    for i in np.arange(0, len(x_set)):
        y_pred[i] = model.predict(np.asarray(x_set[i]).reshape(1, len(x_set[i])))

    predictions = np.zeros(y_pred.shape)
    predictions[y_pred >= 0.5] = 1  # convert to zeros and ones

    acc = jaccard_similarity_score(y_set, predictions)
    f1_micro = f1_score(y_set, predictions, average='micro')
    f1_macro = f1_score(y_set, predictions, average='macro')

    print("Testing Acc: %.3f .. f1_micro: %s ... f1_macro: %s" %(acc, f1_micro, f1_macro))


#######################################################

BATCH_SIZE =32
EMBEDDING_VECTOR_SIZE = 300
HIDDEN_SIZE = 192
EPOCHS = 24
NUMBER_CLASSES = 11 #emotion classes

labels_index = {'anger': 0,
                'anticipation': 1,
                'disgust': 2,
                'fear': 3,
                'joy': 4,
                'love': 5,
                'optimism': 6,
                'pessimism': 7,
                'sadness': 8,
                'surprise': 9,
                'trust': 10} # dictionary mapping label name to numeric id


print("loading and processing text dataset ....")

training_set = [] # list of text samples
training_label = [] # list of label ids
testing_set = []
testing_label = []

training_data, training_label = read_data_from_file("Datasets/Emotion_trainingSet.txt", number_of_classes=NUMBER_CLASSES)
testing_data, testing_label = read_data_from_file("Datasets/Emotion_testingSet.txt", number_of_classes=NUMBER_CLASSES)



print("Converting to sequence & building embedding matrixx...")

dataset = []
dataset.extend(training_data)
dataset.extend(testing_data)

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(filters='"#$%&()*+,-./:;<=>@[\]^_`{|}~ ', oov_token='UNK')
tokenizer.fit_on_texts(dataset)
dataset = tokenizer.texts_to_sequences(dataset)

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

training_set = np.asarray(dataset[:len(training_data)])
testing_set = np.asarray(dataset[-len(testing_data):])

training_label = np.asarray(training_label)
testing_label = np.asarray(testing_label)

#validation_set = training_set[-585:]
#validation_label = training_label[-585:]

#training_set = training_set[:-585]
#training_label = training_label[:-585]

indices = np.arange(len(training_set))
np.random.shuffle(indices)

training_set = training_set[indices] #shuffling data
training_label = training_label[indices]

#model.add(SimpleRNN(HIDDEN_SIZE, dropout=0.2, recurrent_dropout=0.2))
print('Training model...')

model = Sequential()
model.add(Embedding(len(word_index)+1,
                    EMBEDDING_VECTOR_SIZE,
                    weights=[embedding_matrix],
                    trainable=False))

#model.add(GRU(HIDDEN_SIZE, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(GRU(HIDDEN_SIZE, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(NUMBER_CLASSES, activation='sigmoid')) #11 emotions

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#training
fit(model, training_set, training_label, BATCH_SIZE, testing_set, testing_label) #should be implemented using callbacks

#testing
#evaluate(model, testing_set, testing_label)


'''

               Batch 32, gru 128
               12      15      17      20     24
                      0.514   0.520*   0.515

               Batch 64, gru 128
               12      15      17      20     24
                                              0.513

               Batch 96, gru 128
               12      15      17      20     24
               --      --      --      --     --

                Batch 128, gru 128
               10      15      17      20     24
               --      --      --      --     --

            -----------------------------
               Batch 32, lstm 128
               10      12      15      17     20
                                              0.516

               Batch 64, lstm 128
               10      12      15      17     20
                                              0.516

               Batch 96, lstm 128
               10      12      15      17     20
                                              0.512


               Batch 128, lstm 128
               10      12      15      17     20

            ---------------------------

                Batch 32, GRU 160
               12      15      17      20     24
                               0.524

                Batch 32, GRU 192
               12      15      17      20     24
                               0.527  0.524


               Batch 32, GRU 200
               12      15      17      20     24
                               0.526   0.531*

               Batch 32, LSTM 160
                                             0.521

testing:
    0.534


'''