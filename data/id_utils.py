import json


def load_id_stopwords(path='/home/luthfi/workspace/unsupervised-paraphrase-generation/id_stopwords.txt'):
    with open(path) as fin:
        stopwords = fin.read().split('\n')
    return stopwords


def load_thesaurus(path='/home/luthfi/workspace/unsupervised-paraphrase-generation/data/thesaurus.json'):
    with open(path) as fin:
        return json.loads(fin.read())


thesaurus = load_thesaurus()
def get_synonyms(word):
    entry = thesaurus.get(word)
    if entry:
        return entry['sinonim']
    return []

if __name__ == '__main__':
    stopwords = load_id_stopwords()
    print(stopwords)

    # thesaurus = load_thesaurus()
    # print(thesaurus)

    word = 'makan'
    synonyms = get_synonyms(word)
    print(synonyms)