import numpy as np
import pandas as pd
import json

from asl_data import AslDb, SinglesData
from my_model_selectors import (
    SelectorConstant, SelectorBIC,
    SelectorDIC, SelectorCV, ModelSelector
)    
from my_recognizer import recognize

def wer(guesses: list, test_set: SinglesData):
    """Calculate WER
    :param guesses: list of word prediction, ordered
    :param test_set: SinglesData object
    :return: word error rate
    """
    num_test_words = len(test_set.wordlist)
    if len(guesses) != num_test_words:
        raise ValueError()
    count = sum([guesses[i] != test_set.wordlist[i] for i in range(num_test_words)])
    return float(count) / num_test_words

def train_all_words(features, model_selector, asl):
    training = asl.build_training(features)
    sequences = training.get_all_sequences()
    Xlengths = training.get_all_Xlengths()
    model_dict = {}
    for word in training.words:
        model = model_selector(sequences, Xlengths, word,
                               n_constant=3).select()
        model_dict[word]=model
    return model_dict
    
def experiment(feature_list: list, selector: ModelSelector, asl: AslDb):
    """Train model and return WER"""
    models = train_all_words(feature_list, selector, asl)
    test_set = asl.build_test(feature_list)
    _, guesses = recognize(models, test_set)
    return wer(guesses, test_set)

def main():
    selectors = {'bic': SelectorBIC,
                 'dic': SelectorDIC,
                 'cv': SelectorCV}

    base_features = ['right-x', 'right-y','left-x', 'left-y']
    features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']
    features_polar = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']
    features_norm = ['norm-rx', 'norm-ry', 'norm-lx','norm-ly']
    features_delta = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']
    features_deltanorm = ['dn-rx', 'dn-ry', 'dn-lx','dn-ly']
    features_polarnorm = ['polar-norm-rr', 'polar-norm-lr',
                          'polar-rtheta', 'polar-ltheta']
    features = {'ground': features_ground, 'polar': features_polar,
                'norm': features_norm, 'delta': features_delta,
                'deltanorm': features_deltanorm,
                'polarnorm': features_polarnorm}
    # Initiate DB
    asl = AslDb()
    # add features
    print('Adding features')
    asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
    asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
    asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
    asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']
    
    zscore = lambda x: (x - x.mean()) / x.std()
    nrml = asl.df.groupby('speaker')[features_ground].transform(zscore)
    nrml.columns = features_norm
    asl.df = asl.df.join(nrml)
    
    asl.df['polar-rr'] = (asl.df['grnd-rx'] ** 2 + asl.df['grnd-ry'] ** 2) ** 0.5
    asl.df['polar-lr'] = (asl.df['grnd-lx'] ** 2 + asl.df['grnd-ly'] ** 2) ** 0.5
    asl.df['polar-rtheta'] = np.arctan2(asl.df['grnd-rx'], asl.df['grnd-ry'])
    asl.df['polar-ltheta'] = np.arctan2(asl.df['grnd-lx'], asl.df['grnd-ly'])
    
    for i in range(len(base_features)):
        asl.df[features_delta[i]] = asl.df[base_features[i]].diff().fillna(0)
        
    zscore = lambda x: (x - x.mean()) / x.std()
    delta_nrml = asl.df.groupby('speaker')[features_delta].transform(zscore)
    delta_nrml.columns = features_deltanorm
    asl.df = asl.df.join(delta_nrml)

    polar_rnorm = asl.df.groupby('speaker')[['polar-rr', 'polar-lr']].transform(zscore)
    polar_rnorm.columns = ['polar-norm-rr', 'polar-norm-lr']
    asl.df = asl.df.join(polar_rnorm)

    results = {}
    try:
        for s_name, s in selectors.items():
            print('Start ' + s_name)
            results[s_name] = {f_name: experiment(f, s, asl)
                               for f_name, f in features.items()}
            print(results)
    except Exception as e:
        print(e)
    with open('recognizer_result_cv.json', 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    main()
    
