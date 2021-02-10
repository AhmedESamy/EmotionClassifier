import re
import gensim
import  numpy as np
import keras as k

from nltk.stem.isri import ISRIStemmer
from sklearn.model_selection import StratifiedKFold

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


def fit(model, TopicModel, training_set, training_label, BATCH_SIZE, testing_set, testing_label, EPOCHS):

    Number_Batches = int(len(training_set) / BATCH_SIZE)
    records_covered = Number_Batches * BATCH_SIZE
    extra_batch = 0  # in case records didn't fit all in batches

    result = []

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
        #print("Epoch: %s/%s ... Loss: %.2f ... Acc: %.3f" % ((epoch + 1), EPOCHS, score[0], score[1]))

        if ((epoch + 1) in [15, 17, 20, 24]):
            acc, _ , _ = evaluate(model, TopicModel, testing_set, testing_label)
            result.append(acc)

    return result


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

    return acc, f1_micro, f1_macro


#######################################################
BATCH_SIZE = 64
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


#indices = np.arange(len(training_set))
#np.random.shuffle(indices)

#training_set = training_set[indices] #shuffling data
#training_label = training_label[indices]


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

def create_model():

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

    return model
'''
results = []
skf = StratifiedKFold(n_splits=5)
y_dummy = np.zeros(len(training_label)) #skf takes 1D y and splits according to that

split = 1
for train_index, test_index in skf.split(training_set, y_dummy): #loop for cross-validation
    model = create_model()

    xtrain, ytrain = training_set[train_index], training_label[train_index]  # training set
    x_valid, y_valid = training_set[test_index], training_label[test_index] #validation set

    print("split: ", split)
    accuracies = fit(model, topicModel, xtrain, ytrain, BATCH_SIZE, x_valid, y_valid, EPOCHS)  # should be implemented using callbacks

    results.append(accuracies)
    split +=1


result = np.mean(results, axis=0) #get the final average of all accuracies from the k-fold CV
print(result)
'''


#training
model = create_model()
fit(model, topicModel, training_set, training_label, BATCH_SIZE, testing_set, testing_label, EPOCHS) #should be implemented using callbacks

#testing
#evaluate(model, topicModel, testing_set, testing_label)




'''

               Batch 32, gru 128 (343,345)
               15           17           20           24
               [0.53296013 0.5323842  0.53363581 0.53607371]
               [0.53650389 0.53453291 0.53420817 0.53528029]
               [0.53457064 0.5326724  0.53318059 0.5350028 ]
               [0.53199356 0.53074537 0.53236282 0.53236179]

               0.535       0.533      0.534      0.5345*


               Batch 64, gru 128 (346,348)
               15           17           20           24
               [0.52447927 0.52887596 0.52895546 0.53021271]


               Batch 96, gru 128 (249,251)
               15           17           20           24
               [0.52289207 0.52355162 0.52924697 0.53523278]
               [0.51959177 0.52034929 0.52771861 0.53436067]
               [0.51950526 0.52347782 0.52802992 0.53938231]
               [0.51971512 0.52381426 0.52875691 0.53314833]

                                                 0.5353*

                Batch 128, gru 128
               10      15      17      20     24
               --      --      --      --     --

            -----------------------------
               Batch 32, lstm 128
               15           17           20           24
               [0.53321867 0.53429705 0.53692538 0.53198073]
               [0.53477591 0.53050628 0.53641965 0.52984348]
               [0.53191951 0.53155987 0.53321835 0.53261659]
               [0.5303106  0.529457   0.53410054 0.53127096]

                                      0.5351*


               Batch 64, lstm 128
               15           17           20           24
               [0.52801807 0.52999373 0.53036992 0.53097324]
               [0.5259879  0.52839    0.52680925 0.53101999]


               Batch 96, lstm 128
               10      12      15      17     20

            av --      --     --       --      --

               Batch 128, lstm 128
               10      12      15      17     20

            av --      --     --       --      --

            ---------------------------

                Batch 32, GRU 160
                15           17           20           24
                [0.53345392* 0.52936277 0.53218775 0.52523865]
                [0.53310959 0.53575346* 0.53463258 0.52515464]
                [0.53280288 0.53319991 0.53081581 0.52418697]

                0.533       0.5326      0.5326


                Batch 32, GRU 192
                15           17           20           24
                [0.53162649 0.52820528 0.5254513  0.52334652]
                [0.53311749 0.52617424 0.52414975 0.52645343]

               Batch 96, GRU 96
               --      --     --       --      --

               Batch 96, GRU 160
               --      --     --       --      --






'''