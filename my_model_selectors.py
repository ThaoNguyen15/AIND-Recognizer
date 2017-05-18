import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences

class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    N: number of word sequences
    p: number of parameters (which is #states x 3 - escape_p, mean, variance)
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        log_samples = math.log(len(self.X))
        states = range(self.min_n_components, self.max_n_components+1)
        models = []
        logLs = []
        for s in states:
            try:
                model = GaussianHMM(n_components=s, covariance_type='diag',
                                    n_iter=1000, random_state=self.random_state,
                                    verbose=False).fit(self.X, self.lengths)
                models.append(model)
            except:
                models.append(None)
            try:
                logLs.append(model.score(self.X, self.lengths))
            except:
                logLs.append(float('-inf'))
        logLs = np.array(logLs)
        # Calculate the number of free parameters
        # p = n_components**2 + 2 * len(self.X[0]) * n_components - 1
        dimensions = len(self.X[0])
        p = np.array([s**2 + 2*dimensions*s - 1 for s in states])
        bics = - 2 * logLs + p * log_samples
        best_bic_id = np.argmin(bics)
        return models[best_bic_id]


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        states = range(self.min_n_components, self.max_n_components+1)
        models = []
        logLs = []
        rest_logLs = []
        for s in states:
            try:
                model = GaussianHMM(n_components=s, covariance_type='diag',
                                    n_iter=1000, random_state=self.random_state,
                                    verbose=False).fit(self.X, self.lengths)
            except:
                model = None
            models.append(model)
            try:
                logLs.append(model.score(self.X, self.lengths))
                rest_logLs.append(sum([model.score(value[0], value[1]) / len(value[1])
                                       for key, value in self.hwords.items()
#                                       if key == 'CHOCOLATE']))                                       
                                       if key != self.this_word]))
            except:
                logLs.append(float('-inf'))
                rest_logLs.append(float('+inf'))
        num_words = len(self.hwords) - 1
        logLs = np.array(logLs) / len(self.lengths)
        rest_logLs = np.array(rest_logLs)
        result = logLs -  rest_logLs / num_words
        return models[np.argmax(result)]


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        states = range(self.min_n_components, self.max_n_components+1)
        num_samples = len(self.lengths)
        if num_samples <= 1:
            return self.base_model(3)
        elif num_samples == 2:
            num_folds = 2
        else:
            num_folds = 3
        folds = KFold(n_splits=num_folds)
        logLs = np.zeros([num_folds,
                          self.max_n_components + 1 - self.min_n_components])
        for fid, idx_pairs in enumerate(folds.split(self.lengths)):
            train_idx, test_idx = idx_pairs
            X_train, l_train = combine_sequences(train_idx, self.sequences)
            X_test, l_test = combine_sequences(test_idx, self.sequences)
            for sid, s in enumerate(states):
                try:
                    model = GaussianHMM(n_components=s, covariance_type='diag',
                                        n_iter=1000, random_state=self.random_state,
                                        verbose=False).fit(X_train, l_train)
                    logLs[fid][sid] = model.score(X_test, l_test)
                except:
                    # return self.base_model(self.min_n_components)
                    logLs[fid][sid] = float('-inf')
        best_num_states = self.min_n_components + np.argmax(logLs.sum(axis=0))
        return self.base_model(best_num_states)
