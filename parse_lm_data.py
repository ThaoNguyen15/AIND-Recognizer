from os.path import join
import json

def parse():
    with open(join('data', 'ukn.3.lm'), 'r') as f:
        while True:
            l = f.readline()
            if not l:
                break
            if l.strip() == "\data\\":
                # read the next 3 lines
                ngram_counts = []
                for i in range(3):
                    l = f.readline().strip()
                    _, count = l.split('=')
                    ngram_counts.append(int(count))
                print(ngram_counts)
            elif l.strip() == "\\1-grams:":
                unigram = {}
                for i in range(ngram_counts[0]):
                    l = f.readline().strip()
                    pLog, word = l.split('\t')[:2]
                    unigram[word.strip()] = float(pLog)
            elif l.strip() == '\\2-grams:':
                bigram = {}
                for i in range(ngram_counts[1]):
                    l = f.readline().strip()
                    pLog, pair = l.split('\t')[:2]
                    w1, w2 = pair.strip().split()
                    bigram[' '.join([w1.strip(), w2.strip()])] = float(pLog)
            elif l.strip() == '\\3-grams:':
                trigram = {}
                for i in range(ngram_counts[2]):
                    l = f.readline().strip()
                    pLog, pair = l.split('\t')[:2]
                    w1, w2, w3 = pair.strip().split()
                    trigram[' '.join([w1.strip(), w2.strip(),
                                      w3.strip()])] = float(pLog)
        return {'uni': unigram, 'bi': bigram, 'tri': trigram}
                
if __name__ == '__main__':
    d =  parse()
    with open('data/slm.json', 'w') as f:
        json.dump(d, f)
