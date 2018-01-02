import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    for word_id, Xl in test_set.get_all_Xlengths().items():
        X, l = Xl
        prob = {}
        for w, m in models.items():
            try:
                prob[w] = m.score(X, l)
            except:
                prob[w] = float('-inf')
        #prob = {w: m.score(X, l) for w, m in models.items()}
        probabilities.append(prob)
        guesses.append(max(prob, key=lambda x: prob[x]))
    return probabilities, guesses

def hmm_slm_combo(slm_weight, hmm_probabilities):
    pass
