import numpy as np
import keras
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional
import os

dir_path = os.path.dirname(os.path.realpath(__file__))


class BaseTextClassifier(object):
    def train(self, X, y):
        """
        Training using data X, y
        :param X: input feature array
        :param y: label array
        :return:
        """
        pass

    def predict(self, X):
        """
        Predict for input X
        :param X: input feature array
        :return: y as a label array
        """
        pass


def load_synonym_dict(file_path):
    """
    Load synonyms from file and create synonym dictionary
    :param file_path: path to synonym file
    :return: synonym dictionary
    """
    sym_dict = dict()
    with open(file_path, 'r',encoding="utf8") as fr:
        lines = fr.readlines()
    lines = [ln.strip() for ln in lines if len(ln.strip()) > 0]
    for ln in lines:
        words = ln.split(",")
        words = [w.strip() for w in words]
        print(words[1:])
        for word in words[1:]:
            sym_dict.update({word: words[0]})
        print(sym_dict)
    return sym_dict

def load_data_from_file2(file_path):
    """
    Method to load sentences from file
    :param file_path: path to file
    :return: list of sentences
        """
    with open(file_path, 'r',encoding="utf8") as fr:
        sentences = fr.readlines()
        sentences = [sent.strip() for sent in sentences if len(sent.strip()) > 0]
    return sentences


class KerasTextClassifier(BaseTextClassifier):
    def __init__(self, tokenizer, word2vec, model_path, max_length=20, n_epochs=15, batch_size=6,
                 n_class=3, sym_dict=None):
        """
        Create Text Classifier which based on Keras
        :param tokenizer: tokenizer to do correct word segmentation
        :param word2vec: word2vec dictionary, convert word to vector
        :param model_path: path to save or load model
        :param max_length: max length of a sentence
        :param n_epochs: number of epochs
        :param batch_size: number of samples in each batch
        :param n_class: number of classes
        :param sym_dict: synonym dictionary (optional)
        """
        self.tokenizer = tokenizer
        self.word2vec = word2vec
        print(self.word2vec[self.word2vec.index2word[0]].shape[0])
        self.word_dim = self.word2vec[self.word2vec.index2word[0]].shape[0]
        self.max_length = max_length
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.model_path = model_path
        self.n_class = n_class
        self.model = None
        self.sym_dict = sym_dict

    def train(self, X, y):
        """
        Training with data X, y
        :param X: 3D features array, number of samples x max length x word dimension
        :param y: 2D labels array, number of samples x number of class
        :return:
        """
        print("have train")
        self.model = self.build_model(input_dim=(X.shape[1], X.shape[2]))
        self.model.fit(X, y, batch_size=self.batch_size, epochs=self.n_epochs)
        self.model.save_weights(self.model_path)

    def predict(self, X):
        """
        Predict for 3D feature array
        :param X: 3D feature array, converted from string to matrix
        :return: label array y as 2D-array
        """
        if self.model is None:
            self.load_model()
        y = self.model.predict(X)
        return y

    def classify(self, sentences, label_dict=None):
        """
        Classify sentences
        :param sentences: input sentences in format of list of strings
        :param label_dict: dictionary of label ids and names
        :return: label array
        """
        print(sentences)
        X = [sent.strip() for sent in sentences]
        X, _ = self.tokenize_sentences(X)
        X = self.word_embed_sentences(X, max_length=self.max_length)
        print(X)
        y = self.predict(np.array(X))
        print(y)
        y = np.argmax(y, axis=1)
        labels = []
        for lab_ in y:
            if label_dict is None:
                labels.append(lab_)
            else:
                labels.append(label_dict[lab_])
        
        return labels

    def load_model(self):
        """
        Load model from file
        :return: None
        """
        self.model = self.build_model((self.max_length, self.word_dim))
        self.model.load_weights(self.model_path)

    def build_model(self, input_dim):
        """
        Build model structure
        :param input_dim: input dimension max_length x word_dim
        :return: Keras model
        """
        model = Sequential()

        model.add(LSTM(64, return_sequences=True, input_shape=input_dim))
        model.add(Dropout(0.2))
        model.add(LSTM(32))
        model.add(Dense(self.n_class, activation="softmax"))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(),
                      metrics=tensorflow.keras.metrics.categorical_accuracy)
        return model

    def load_data(self, path_list, load_method):
        """
        Load data from list of paths
        :param path_list: list of paths to files or directories
        :param load_method: method to load (from file or from directory)
        :return: 3D-array X and 2D-array y
        """
        X = None
        y = None
        for i, data_path in enumerate(path_list):
            sentences_ = load_method(data_path)
            label_vec = [0.0 for _ in range(0, self.n_class)]
            label_vec[i] = 1.0
            labels_ = [label_vec for _ in range(0, len(sentences_))]
            if X is None:
                X = sentences_
                y = labels_
            else:
                X += sentences_
                y += labels_
            # print(X)
            # print(y)
            # print("-"*50)
        X, max_length = self.tokenize_sentences(X)
        X = self.word_embed_sentences(X, max_length=self.max_length)
        return np.array(X), np.array(y)

    def word_embed_sentences(self, sentences, max_length=20):
        """
        Helper method to convert word to vector
        :param sentences: input sentences in list of strings format
        :param max_length: max length of sentence you want to keep, pad more or cut off
        :return: embedded sentences as a 3D-array
        """
        embed_sentences = []
        print(self.sym_dict)
        for sent in sentences:
            embed_sent = []
            for word in sent:
                if (self.sym_dict is not None) and (word.lower() in self.sym_dict):
                    # print("1: ",word)
                    replace_word = self.sym_dict[word.lower()]
                    embed_sent.append(self.word2vec[replace_word])
                elif word.lower() in self.word2vec:
                    # print("2: ", word)
                    embed_sent.append(self.word2vec[word.lower()])
                else:
                    # print("3: ", word)
                    embed_sent.append(np.zeros(shape=(self.word_dim,), dtype=float))
            if len(embed_sent) > max_length:
                embed_sent = embed_sent[:max_length]
            elif len(embed_sent) < max_length:
                embed_sent = np.concatenate((embed_sent, np.zeros(shape=(max_length - len(embed_sent),
                                                                         self.word_dim), dtype=float)),
                                            axis=0)
            embed_sentences.append(embed_sent)
        return embed_sentences

    def tokenize_sentences(self, sentences):
        """
        Tokenize or word segment sentences
        :param sentences: input sentences
        :return: tokenized sentence
        """
        tokens_list = []
        max_length = -1
        for sent in sentences:
            tokens = self.tokenizer.tokenize(sent)
            tokens_list.append(tokens)
            if len(tokens) > max_length:
                max_length = len(tokens)
        # print(tokens_list) # tokens_list is array 2D
        return tokens_list, max_length

    @staticmethod
    def load_data_from_file(file_path):
        """
        Method to load sentences from file
        :param file_path: path to file
        :return: list of sentences
        """
        with open(file_path, 'r',encoding="utf8") as fr:
            sentences = fr.readlines()
            sentences = [sent.strip() for sent in sentences if len(sent.strip()) > 0]
        return sentences


class BiDirectionalLSTMClassifier(KerasTextClassifier):
    def build_model(self, input_dim):
        """
        Overwrite build model using Bidirectional Layer
        :param input_dim: input dimension
        :return: Keras model
        """
        model = tensorflow.keras.Sequential()

        model.add(Bidirectional(LSTM(32, return_sequences=True), input_shape=input_dim))
        model.add(Dropout(0.1))
        model.add(Bidirectional(LSTM(16)))
        model.add(Dense(self.n_class, activation="softmax"))

        model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
                      optimizer=tensorflow.keras.optimizers.Adam(),
                      metrics=tensorflow.keras.metrics.categorical_accuracy)
        return model



"""Tests"""


def test():
    from api.tokenization.dict_models import LongMatchingTokenizer
    from api.word_embedding.word2vec_gensim import Word2Vec
    word2vec_model = Word2Vec.load(dir_path + '/../models/word2vec.LM.model')
    # Please give the correct paths
    tokenizer = LongMatchingTokenizer()
    sym_dict = load_synonym_dict(dir_path + '/../data/sentiment/synonym.txt')
    keras_text_classifier = BiDirectionalLSTMClassifier(tokenizer=tokenizer, word2vec=word2vec_model.wv,
                                                        model_path=dir_path + '/../models/sentiment_model_5.h5',
                                                        max_length=200, n_epochs=10,
                                                        sym_dict=sym_dict)

    

    label_dict = {0: 'tích cực', 1: 'tiêu cực', 2: 'bình thường'}

    test_sentences = load_data_from_file2(dir_path+'/../data/sentiment/testpos.txt')
    labels = keras_text_classifier.classify(test_sentences, label_dict=label_dict)
    print(labels)
    S=0
    for x in labels: 
        if x == "tích cực":
            S+=1
    print(S)
    print(len(labels))
    ratio=S/len(labels)
    print(ratio)

if __name__ == '__main__':
    test()
