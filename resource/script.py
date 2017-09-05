CORPUS_PATH = 'corpus.txt'


def build_corpus():
    with open(CORPUS_PATH, 'r') as f:
        with open('source.txt', 'w') as source:
            with open('target.txt', 'w') as target:
                s_arr = []
                t_arr = []
                for line in f.readlines():
                    line = line.strip()
                    if line != '':
                        s, t = line.split()
                        s_arr.append(s)
                        t_arr.append(t)
                    else:
                        source.write(' '.join(s_arr) + '\n')
                        target.write(' '.join(t_arr) + '\n')
                        s_arr = []
                        t_arr = []


def build_word_index():
    with open('source.txt', 'r') as source:
        dict_word = {}
        # with open('source_vocab', 'w') as s_vocab:
        for line in source.readlines():
            line = line.strip()
            if line != '':
                word_arr = line.split()
                for w in word_arr:
                    dict_word[w] = dict_word.get(w, 0) + 1

        top_words = sorted(dict_word.items(), key=lambda s: s[1], reverse=True)
        with open('source_vocab.txt', 'w') as s_vocab:
            for word, frequence in top_words:
                s_vocab.write(word + '\n')

    with open('target.txt', 'r') as source:
        dict_word = {}
        # with open('source_vocab', 'w') as s_vocab:
        for line in source.readlines():
            line = line.strip()
            if line != '':
                word_arr = line.split()
                for w in word_arr:
                    dict_word[w] = dict_word.get(w, 0) + 1

        top_words = sorted(dict_word.items(), key=lambda s: s[1], reverse=True)
        with open('target_vocab.txt', 'w') as s_vocab:
            for word, frequence in top_words:
                s_vocab.write(word + '\n')


build_word_index()