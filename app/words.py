import os
from nltk.stem import PorterStemmer

from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()

FILES_FOLDER = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), os.pardir, "files")

WORDS_FILE = os.path.join(FILES_FOLDER, "scowl10to70.txt")

with open(WORDS_FILE, 'r') as file:
    words = file.readlines()

stemmed_words = set([wnl.lemmatize(word.strip())
                    for word in words if len(word) > 0])
sorted_words = sorted(stemmed_words)
with open(os.path.join(FILES_FOLDER, 'scowl10to70_lemmatization.txt'), 'w') as file:
    file.write('\n'.join(sorted_words))
