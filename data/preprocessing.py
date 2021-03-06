import argparse
import csv
import random

from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from transformers import GPT2Tokenizer

from eda import synonym_replacement
from id_utils import load_id_stopwords
import tqdm

english_stopwords = stopwords.words('english')

# Stopwords from case study of the paper
# 1. From case study
english_stopwords += ['someone', 'something', 'make', 'see']
# 2. From possible candidates
english_stopwords += ['everything']
# 3. Similar words from those of case study
english_stopwords += ['anyone', 'anything', 'everyone']

id_stopwords = load_id_stopwords()
stopwords = id_stopwords
tokenizer = TreebankWordTokenizer()
detokenizer = TreebankWordDetokenizer()


def remove_stopwords(sentence):
    sentence = tokenizer.tokenize(sentence)
    sentence = [word for word in sentence
                if word.lower() not in stopwords]
    sentence = ' '.join(sentence)
    sentence = sentence.replace("''", '"').replace('``', '"')
    sentence = detokenizer.detokenize(sentence.split())
    return sentence


def sentence_noising(sentence, shuffle_ratio=0.2, replace_ratio=0.2):
    # 1. Synonym replacement
    words = sentence.split()
    n_sr = max(1, int(len(words)*shuffle_ratio))
    words = synonym_replacement(words, n_sr)

    # 2. Random shuffling
    if random.random() < shuffle_ratio:
        random.shuffle(words)

    return ' '.join(words)


# def data_preparation(args):
#     gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
#     data = []
#     with open(args.input) as f:
#         skipped = 0
#         for line in tqdm.tqdm(f):
#             sentence = line.strip()
#             corrupted_sentence = remove_stopwords(sentence)
#             write_line = corrupted_sentence + '\n' + sentence
#             if len(gpt_tokenizer.encode(write_line)) < args.max_length:
#                 data.append([corrupted_sentence, sentence])
#             else:
#                 skipped += 1
#     print("Skipped: {}".format(skipped))

#     with open(args.output, 'w') as wf:
#         writer = csv.writer(wf)
#         for corrupted, sentence in data:
#             writer.writerow([corrupted, sentence])

#     if args.save_noised_output is True:
#         with open(args.noised_output, 'w') as wf:
#             writer = csv.writer(wf)
#             for corrupted, sentence in data:
#                 corrupted = sentence_noising(corrupted)
#                 writer.writerow([corrupted, sentence])

def data_preparation(args):
    gpt_tokenizer = GPT2Tokenizer.from_pretrained('cahya/gpt2-small-indonesian-522M')
    # data = []

    fout = open(args.output, 'w')
    fout_c = open(args.noised_output, 'w')
    writer1 = csv.writer(fout)
    writer2 = csv.writer(fout_c)

    with open(args.input) as f:
        skipped = 0
        for line in tqdm.tqdm(f):
            sentence = line.strip()
            corrupted_sentence = remove_stopwords(sentence)
            write_line = corrupted_sentence + '\n' + sentence
            if len(gpt_tokenizer.encode(write_line)) < args.max_length:
                writer1.writerow([corrupted_sentence, sentence])
                if args.save_noised_output is True:
                    corrupted_sentence = sentence_noising(corrupted_sentence)
                    writer2.writerow([corrupted_sentence, sentence])
            else:
                skipped += 1
    print("Skipped: {}".format(skipped))



    # with open(args.output, 'w') as wf:
    #     writer = csv.writer(wf)
    #     for corrupted, sentence in data:
    #         writer.writerow([corrupted, sentence])

    # if args.save_noised_output is True:
    #     with open(args.noised_output, 'w') as wf:
    #         writer = csv.writer(wf)
    #         for corrupted, sentence in data:
    #             corrupted = sentence_noising(corrupted)
    #             writer.writerow([corrupted, sentence])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help='input file')
    parser.add_argument('--output', type=str, required=True,
                        help='output sentence after removing stop words')

    parser.add_argument('--save_noised_output', action="store_true",
                        help='add noise: synonym replacement and shuffling')
    parser.add_argument('--noised_output', type=str, default=None,
                        help='output sentences with additional noise')

    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=1234)

    args = parser.parse_args()

    random.seed(args.seed)

    if args.noised_output is None:
        args.noised_output = args.output + '.0'

    data_preparation(args)
