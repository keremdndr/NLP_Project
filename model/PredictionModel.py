import datetime
import pickle

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score

print(datetime.datetime.now().__str__() +" : ************ Start **************")

x_train = "/Users/keremdundar/Desktop/Kerem/Dersler/Bitirme_Projesi/Data/MergeByLabel/x_train.pkl"
vectorFile = "/Users/keremdundar/Desktop/Kerem/Dersler/Bitirme_Projesi/Data/MergeByLabel/vectorizer.pickle"
modelFile = "/Users/keremdundar/Desktop/Kerem/Dersler/Bitirme_Projesi/Data/MergeByLabel/model.sav"
filepath_dict = {'merge':   '/Users/keremdundar/Desktop/Kerem/Dersler/Bitirme_Projesi/Data/MergeByLabel/merge.txt'}
filepath_dict_new = {'newFilm':   '/Users/keremdundar/Desktop/Kerem/Dersler/Bitirme_Projesi/Data/MergeByLabel/newFilm.txt'}

ScriptLabelArray = ["action","adventure","animation","biography","comedy","crime","drama","family","fantasy","film_noir",
                    "history","horror","music","musical","mystery","romance","sci_fi","short","sport","thriller","war",
                    "western"]

df_list = []
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
    df['source'] = source  # Add another column filled with the source name
    df_list.append(df)
print(datetime.datetime.now().__str__() +" : ************ 1.file read **************")

df_list_new = []
for source_new, filepath in filepath_dict_new.items():
    df_new = pd.read_csv(filepath, names=['sentence'])
    df_new['source'] = source_new
    df_list_new.append(df_new)
print(datetime.datetime.now().__str__() +" : ************ 2.file read **************")

df_source = df[df['source'] == 'merge']
sentences = df_source['sentence'].values
y = df_source['label'].values
print(datetime.datetime.now().__str__() +" : ************ 1.file dataFrame transfered **************")


df_source_new = df_new[df_new['source'] == 'newFilm']
sentences_new = df_source_new['sentence'].values
print(datetime.datetime.now().__str__() +" : ************ 2.file dataFrame transfered **************")


# sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.10, random_state=1000)
print(datetime.datetime.now().__str__() +" : ************ vector start **************")

# vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
# print(datetime.datetime.now().__str__() +" : ************ vector start 1 **************")
#
# vectorizer.fit(sentences)
# print(datetime.datetime.now().__str__() +" : ************ vector start 2 **************")
print(datetime.datetime.now().__str__() +" : ************ vector loading **************")
vectorizer = pickle.load(open(vectorFile,"rb"))
print(datetime.datetime.now().__str__() +" : ************ vector loaded **************")

X_train = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open(x_train, "rb")))

# X_train = vectorizer.transform(sentences)
print(datetime.datetime.now().__str__() +" : ************ vector start 3 **************")

X_test = vectorizer.transform(sentences_new)
print(datetime.datetime.now().__str__() +" : ************ vector done **************")
print(datetime.datetime.now().__str__() +" : ************ model loaded **************")

loaded_model = pickle.load(open(modelFile, 'rb'))

print(datetime.datetime.now().__str__() +" : ************ predection start **************")
label = loaded_model.predict(X_test)
print(datetime.datetime.now().__str__() +" : ************ predection finish **************")

labelAccuracyArray = []

for i in range(0,22):
    print(datetime.datetime.now().__str__() + " : Model make prediction for " + str(i) +".label")
    tempArray = [i] * len(label)
    df_temp = pd.DataFrame(tempArray)
    tempAccuracy = accuracy_score(df_temp,label)
    labelAccuracyArray.append(float(tempAccuracy))
    print(datetime.datetime.now().__str__() + " : Score for " + str(i) + ".label : " + str(labelAccuracyArray[i]))

print("max value is : " + max(labelAccuracyArray).__str__())
max_item_index = labelAccuracyArray.index(max(labelAccuracyArray))
print("Label name is : " + (ScriptLabelArray[max_item_index]).title())