import datetime
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
import random
import pickle

print(datetime.datetime.now().__str__() +" : ************ code start **************")

modelFile = "/Users/keremdundar/Desktop/Kerem/Dersler/Bitirme_Projesi/Data/MergeByLabel/model.sav"
vectorFile = "/Users/keremdundar/Desktop/Kerem/Dersler/Bitirme_Projesi/Data/MergeByLabel/vectorizer.pickle"
x_train = "/Users/keremdundar/Desktop/Kerem/Dersler/Bitirme_Projesi/Data/MergeByLabel/x_train.pkl"
filepath_dict = {'merge' : '/Users/keremdundar/Desktop/Kerem/Dersler/Bitirme_Projesi/Data/MergeByLabel/merge.txt'}
df_list = []

print(datetime.datetime.now().__str__() +" : ************ read file start **************")
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
    df['source'] = source
    df_list.append(df)
print(datetime.datetime.now().__str__() +" : ************ read file finish **************")
# df = pd.concat(df_list)
# print(datetime.datetime.now().__str__() +" : "+ (df.iloc[random.randint(1,100000)]))

print(datetime.datetime.now().__str__() +" : ************ vectorizer start **************")

vectorizer = CountVectorizer(min_df=0, lowercase=False)
vectorizer.fit(df['sentence'])

print(datetime.datetime.now().__str__() +" : ************ vectorized **************")
df_source = df[df['source'] == "merge"]
print('1')
sentences = df_source['sentence'].values
print('2')
labelTrain = df_source['label'].values
print('3')

vectorizer = CountVectorizer(max_features=1500, min_df=25, max_df=0.7, stop_words=stopwords.words('english'))
print('4')
vectorizer.fit(sentences)

print(datetime.datetime.now().__str__() +" : ************ create vector pickle **************")
pickle.dump(vectorizer, open(vectorFile, "wb"))

print(datetime.datetime.now().__str__() +" : ************ 2.vectorizer fit process done **************")
SentencesTrain = vectorizer.transform(sentences)
print(datetime.datetime.now().__str__() +" : ************ 2.vectorizer transform process done **************")

pickle.dump(SentencesTrain, open(x_train, "wb"))
print(datetime.datetime.now().__str__() +" : ************ x_train vectorizer write to file **************")

print(datetime.datetime.now().__str__() +" : ************ 2.vectorizer transform process done **************")

print(datetime.datetime.now().__str__() +": ************ naive bayes **************")
clf = MultinomialNB()
clf.fit(SentencesTrain, labelTrain)

print(datetime.datetime.now().__str__() +": ************ naive bayes done **************")

pickle.dump(clf, open(modelFile, 'wb'))

print(datetime.datetime.now().__str__() +": ************ model saved **************")
print(datetime.datetime.now().__str__() +": ************ code stop **************")