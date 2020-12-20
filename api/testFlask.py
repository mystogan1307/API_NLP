from flask import Flask, jsonify, request
import nltk
from nltk.tokenize import word_tokenize
import os
from flask_cors import CORS, cross_origin

dir_path = os.path.dirname(os.path.realpath(__file__))

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
#@app.route("/")
@cross_origin()


@app.route('/api/analysis', methods=['POST'])
def index():
    request_data = request.get_json()
    comments = request_data['comments']
    print(comments)
    from .tokenization.dict_models import LongMatchingTokenizer
    from .word_embedding.word2vec_gensim import Word2Vec
    from .text_classification.short_text_classifiers import BiDirectionalLSTMClassifier, load_synonym_dict
    keras_text_classifier = None
    word2vec_model = None

    word2vec_model = Word2Vec.load(dir_path + '/models/word2vec.LM.model')


    tokenizer = LongMatchingTokenizer()
    sym_dict = load_synonym_dict(dir_path  + '/data/sentiment/synonym.txt')
    keras_text_classifier = BiDirectionalLSTMClassifier(tokenizer=tokenizer, word2vec=word2vec_model.wv,
                                                        model_path=dir_path + '/models/sentiment_model_5.h5',
                                                        max_length=200, n_epochs=50,
                                                        sym_dict=sym_dict)
 
    label_dict = {0: 'tích cực', 1: 'tiêu cực', 2: 'bình thường'}
    test_sentences = comments
    labels = keras_text_classifier.classify(test_sentences, label_dict=label_dict)
    print(labels)  
    
    return jsonify(labels)




if __name__ == '__main__':
    app.run( port=8000,debug=True)