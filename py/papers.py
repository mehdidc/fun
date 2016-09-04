from gensim import corpora, models, similarities, models
import glob
import re

from cStringIO import StringIO
expr = re.compile("([^\w\xe9-\xf8'\s])", re.UNICODE)

folders = [
        "/home/mehdi/A Lire/info/papers/ML"
]

def to_words(fd):
    lines = []
    for line in fd.readlines():
        line = line.decode("utf8")
        line = line[:-1]
        line = expr.sub(r" \1 ", line, re.UNICODE)
        line = line.lower()
        line = line.split()
        line = [word.strip() for word in line]
        lines.append(line)
    words = [word for line in lines for word in line]
    return words

class MyCorpus(object):
    def __init__(self, filenames):
       self.filenames = filenames 
    def __iter__(self):
        for filename in self.filenames:
            fd = open(filename)
            words = to_words(fd)
            fd.close()
            yield words


import os
import subprocess
import re
def findfiles(dir, pattern):
    patternregex = re.compile(pattern)
    for root, dirs, files in os.walk(dir):
        for basename in files:
            filename = os.path.join(root, basename)
            if patternregex.search(filename, re.IGNORECASE):
                yield filename

def generate_txt_all(folder):
    for filename in findfiles(folder, "\.pdf$"):
        generate_txt(filename)

def generate_txt(filename):
    subprocess.call(["pdftotext", filename], cwd=os.path.join(os.getenv("HOME"), ".papers"))

if __name__ == "__main__":

    import sys
    action = sys.argv[1]
    folder = os.path.join(os.getenv("HOME"), ".papers")
    filenames = list(glob.glob(folder + "/*.txt"))
    filenames = sorted(filenames)

    if action == "gentext":
        for folder in folders:
            generate_txt_all(folder)
    elif action == "buildmodel":
        corpus = MyCorpus(filenames)
        dictionary = corpora.Dictionary(corpus)
        dictionary.save(folder + "/dictionary")
        corpus = [dictionary.doc2bow(doc) for doc in corpus]
        corpora.MmCorpus.serialize(os.path.join(folder, "corpus"), corpus)
        corpus = corpora.MmCorpus(os.path.join(folder, "corpus"))
        model = models.LdaModel(corpus, id2word=dictionary, num_topics=200)
        index = similarities.MatrixSimilarity(model[corpus])
        model.save(folder + "/model")
        index.save(folder + "/index")
    elif action == "query":
        corpus = corpora.MmCorpus(folder + "/corpus")
        model = models.LsiModel.load(folder + "/model")
        index = similarities.MatrixSimilarity.load(folder + "/index")
        dictionary = corpora.Dictionary.load(folder + "/dictionary")
        query = sys.argv[2]
        if len(sys.argv) >= 4:
            top = int(sys.argv[3])
        else:
            top = 1
        query = model[dictionary.doc2bow(to_words(StringIO(query)))]
        response = index[query]
        rank = sorted(range(len(corpus)), key=lambda k:response[k], reverse=True)

        rank_filenames = (map(lambda k:filenames[k], rank))
        response = rank_filenames[top - 1]
        print(response)
        for folder in folders:
            for filename in findfiles(folder, "\.pdf$"):
                if(os.path.basename(filename).split(".")[0] == 
                   os.path.basename(response).split(".")[0]):
                    subprocess.call(["evince", filename])
                    break
