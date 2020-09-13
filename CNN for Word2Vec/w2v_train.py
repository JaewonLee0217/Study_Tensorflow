from gensim.test.utils import get_tmpfile
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence, PathLineSentences

model = Word2Vec.load("word2vec.model")
score, predictions = model.wv.evaluate_word_analogies('./data/questions-words.txt')
print(score)