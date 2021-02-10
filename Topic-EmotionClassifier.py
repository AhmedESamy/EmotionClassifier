import re
import gensim
import  numpy as np
import keras as k

from nltk.stem.isri import ISRIStemmer

from keras.models import Model, Sequential, model_from_json
from keras.layers import Embedding, Dense, Activation, Dropout, GRU, Input,Conv2D, GlobalMaxPool2D, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import f1_score, jaccard_similarity_score

#written by Ahmed Emad May 2019


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

        label = list(map(int, filtered_line[-11:]))

        for word in filtered_line[1:-11]:
            tweet += word + " "

        tweet = tweet[:-1]
        tweet = clean_str(tweet)

        tweet = istemmer.norm(tweet)

        tweets.append(tweet)
        labels.append(label)

    read_file.close()

    return [tweets, labels]


def write_to_file (y_pre):
    read_file = open("testing_set_Ar.txt", "r+")  # read and write mode
    write_test_file = open("/output/E-C_ar_pred.txt", "w")  # write mode

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


def predict_topics_representations(topicModel, x_set):
    return topicModel.predict(x_set)


def fit(model, TopicModel, training_set, training_label, BATCH_SIZE, testing_set, testing_label):

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

            topic_batch_sequences = predict_topics_representations(TopicModel, batch_sequences) #return the training set after batching it.
            topic_batch_sequences = np.asarray(topic_batch_sequences).reshape(len(topic_batch_sequences),
                                                                              len(topic_batch_sequences[0]),
                                                                              len(topic_batch_sequences[0][0]),
                                                                              1)  # 4-D (samples, timestamps, 128-dimensonal_lstm, channel))

            model.train_on_batch([topic_batch_sequences, batch_sequences], [batch_labels, batch_labels])
            score += evaluate_on_batch(model, batch_sequences, batch_labels, topic_batch_sequences)

        if (len(training_set) > records_covered):  # for remaining records
            batch_sequences = np.asarray(training_set[records_covered: len(training_set)])
            batch_sequences = pad_sequences(batch_sequences)  # Pad to maximum length per each batch

            batch_labels = training_label[records_covered: len(training_set)]

            topic_batch_sequences = predict_topics_representations(TopicModel, batch_sequences)  # return the training set after batching it.
            topic_batch_sequences = np.asarray(topic_batch_sequences).reshape(len(topic_batch_sequences),
                                                                              len(topic_batch_sequences[0]),
                                                                              len(topic_batch_sequences[0][0]),
                                                                              1)  # 4-D (samples, timestamps, 128-dimensonal_lstm, channel))

            model.train_on_batch([topic_batch_sequences, batch_sequences], [batch_labels, batch_labels])
            score += evaluate_on_batch(model, batch_sequences, batch_labels, topic_batch_sequences)

            extra_batch = 1

        score[0] /= (Number_Batches + extra_batch)
        score[1] /= (Number_Batches + extra_batch)
        print("Epoch: %s/%s ... Loss: %.2f ... Acc: %.3f" % ((epoch + 1), EPOCHS, score[0], score[1]))

        if ((epoch + 1) in [10, 12, 15, 17, 20, 24]):
            evaluate(model, TopicModel, testing_set, testing_label)


#evaluate
def evaluate_on_batch(model, x_set, y_set, topicFeatures):

    _, y_pred = model.predict([topicFeatures, x_set])

    predictions = np.zeros(y_pred.shape)
    predictions[y_pred >= 0.5] = 1  # convert to zeros and ones

    acc = jaccard_similarity_score(y_set, predictions)
    loss = -y_set * np.log(y_pred + 10**-10) - (1 - y_set) * np.log(1 - y_pred + 10 ** -10)
    total_loss = np.sum(loss) / (y_pred.shape[0] * y_pred.shape[1])

    return [total_loss, acc]


#evaluate
def evaluate(model, topicModel, x_set, y_set):

    y_pred = np.empty(y_set.shape)

    for i in np.arange(0, len(x_set)):

        if len(x_set[i])< 10: #convolutions size is 4.
            x_set[i] = np.pad(x_set[i], (10-len(x_set[i]), 0), 'constant')

        topicFeatures = predict_topics_representations(topicModel, np.asarray(x_set[i]).reshape(1, len(x_set[i])))
        topicFeatures = topicFeatures.reshape(len(topicFeatures),
                                              len(topicFeatures[0]),
                                              len(topicFeatures[0][0]),
                                              1)  # 4-D (samples, timestamps, 128-dimensonal_lstm, channel))

        _, y_pred[i] = model.predict([topicFeatures, np.asarray(x_set[i]).reshape(1, len(x_set[i]))])

    predictions = np.zeros(y_pred.shape)
    predictions[y_pred >= 0.5] = 1  # convert to zeros and ones

    acc = jaccard_similarity_score(y_set, predictions)
    f1_micro = f1_score(y_set, predictions, average='micro')
    f1_macro = f1_score(y_set, predictions, average='macro')

    print("Testing Acc: %.3f .. f1_micro: %s ... f1_macro: %s" %(acc, f1_micro, f1_macro))


#######################################################
BATCH_SIZE = 32
EMBEDDING_VECTOR_SIZE = 300
GRU_HIDDEN_SIZE = 128
MLP_HIDDEN_SIZE = 64
TOPIC_HIDDEN_SIZE = 128
EPOCHS = 24
NUMBER_CLASSES = 11


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
        embedding_matrix[i] = embedding_matrix[1]
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


print("\nExtracting contextual Information (Topics)...")

json_file = open('/TopicModel/topicModel.json','r')
loaded_model_json = json_file.read()
json_file.close()
topic_classifier_model = model_from_json(loaded_model_json)
topic_classifier_model.load_weights("/TopicModel/topicModel.h5")
print("loaded topic model")

topic_classifier_model.compile(optimizer='rmsprop',
                               loss='categorical_crossentropy',
                               metrics=['accuracy'])


# This is because in the topic model, GRU wasn't set to return sequences, only the last gru, and with different embedding dimensons
topicModel = Sequential()
topicModel.add(Embedding(len(word_index)+1,
                             EMBEDDING_VECTOR_SIZE,
                             weights=[embedding_matrix],
                             trainable=False))

topicModel.add(GRU(TOPIC_HIDDEN_SIZE, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
topicModel.compile(optimizer='rmsprop',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])


topicModel.layers[1].set_weights(topic_classifier_model.layers[1].get_weights())


print('Training model...')

topicFeatures = Input(shape=(None,None,1)) #Topic_Input 4-D (samples, timestamps, 128-dimensonal_lstm, channel))

contextualModel = Conv2D(16,(4,4), activation='elu') (topicFeatures)
contextualModel = Conv2D(32,(4,4), activation='elu') (contextualModel)
contextualModel = Dropout(0.2)(contextualModel)
contextualModel = Conv2D(64,(4,4), activation='elu') (contextualModel)
contextualModel = Dropout(0.2)(contextualModel)
contextualModel = Conv2D(128, (1,1))(contextualModel)

encodedContext = GlobalMaxPool2D()(contextualModel)

sen_input = Input(shape=(None,))#Sentence Input

sen_model = Embedding(len(word_index)+1,
                   EMBEDDING_VECTOR_SIZE,
                   weights=[embedding_matrix],
                   trainable=False) (sen_input)


#encoded_sen = GRU(GRU_HIDDEN_SIZE, dropout=0.2, recurrent_dropout=0.2, return_sequences=True) (sen_model) #now vectors from the sentence are converted to one vector
encoded_sen = GRU(GRU_HIDDEN_SIZE, dropout=0.2, recurrent_dropout=0.2) (sen_model) #now vectors from the sentence are converted to one vector


merged_vector = k.layers.concatenate([encodedContext, encoded_sen])
joint_model = Dense(MLP_HIDDEN_SIZE, activation='relu')(merged_vector)
joint_model = Dropout(0.2)(joint_model)

auxiliary_output = Dense(NUMBER_CLASSES, activation='sigmoid') (encodedContext)#12 emotions
main_output = Dense(NUMBER_CLASSES, activation='sigmoid') (joint_model) #12 emotions

model = Model(inputs=[topicFeatures, sen_input], outputs=[auxiliary_output, main_output])

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#training
fit(model, topicModel, training_set, training_label, BATCH_SIZE, testing_set, testing_label) #should be implemented using callbacks

#testing
#evaluate(model, topicModel, testing_set, testing_label)




'''

               Batch 32, gru 128 (343,345)
               12      15      17      20     24
               0.506   0.489   0.517   0.518  0.516
               0.505   0.491   0.515   0.519  0.498
               0.504   0.487   0.505   0.510  0.504

            av                 0.512   0.516* 0.506

               Batch 64, gru 128 (346,348)
               12      15      17      20     24
               0.492   0.490   0.506   0.512  0.511
               0.493   0.498   0.507   0.511  0.511
               0.493   0.492   0.507   0.509  0.510

            av                         0.511  0.511

               Batch 96, gru 128 (249,251)
               12      15      17      20     24
               0.511   0.507   0.502   0.518  0.520
               0.503   0.504   0.500   0.514  0.511
               0.504   0.503   0.500   0.516  0.509

            av                         0.516* 0.513

                Batch 128, gru 128
               10      15      17      20     24
               --      --      --      --     --

            -----------------------------
               Batch 32, lstm 128
               10      12      15      17     20
               0.501   0.508   0.513   0.514  0.512
               0.499   0.510   0.511   0.507  0.507

            av --      --      0.512   --      --

               Batch 64, lstm 128
               10      12      15      17     20
               0.493   0.511   0.495   0.512  0.517
               0.497   0.503   0.491   0.514  0.511
                                              0.513

            av --      --     --       --      --

               Batch 96, lstm 128
               10      12      15      17     20

            av --      --     --       --      --

               Batch 128, lstm 128
               10      12      15      17     20

            av --      --     --       --      --

            ---------------------------

                Batch 32, GRU 160
               12      15      17      20     24
                               0.524          0.500
                               0.517          0.515
               0.504   0.481   0.518          0.511
                               0.512
                               0.515

            av --      --      0.5196*   --

                Batch 32, GRU 192
               12      15      17      20     24
                               0.515          0.517
                               0.511          0.522
                               0.508          0.518
                               0.516          0.520
                                              0.517

            av --      --      --       --    0.5193

               Batch 96, GRU 96

            av --      --     --       --      --

               Batch 96, GRU 160
            av --      --     --       --      --

testing:
        batch 32, GRU 192 (410, 411)
        0.518
        0.526
        0.522

        0.522

        batch 32, GRU 160
        0.529   0.536   0.537   0.4599747291643007
        0.523   0.533   0.540   0.4523091487505446
        0.527   0.530   0.537   0.46103612976487446

        0.526

        batch 32, GRU 128 (412, 413)

        0.534    0.47168138097824736
        0.536
        0.531

        0.534





'''