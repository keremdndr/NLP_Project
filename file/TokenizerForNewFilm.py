import re
from sklearn.datasets import load_files
from nltk.stem.porter import PorterStemmer

porterStemmer = PorterStemmer()

mergedFileName = '/Users/keremdundar/Desktop/Kerem/Dersler/Bitirme_Projesi/Data/MergeByLabel/newFilm.txt'
mergedFile = open(mergedFileName, "a+")



script_data = load_files(r"/Users/keremdundar/Desktop/Kerem/Dersler/Bitirme_Projesi/Data/NewFilm")
data,target = script_data.data, script_data.target
print("asd")
documents = []
for sen in range(0, len(data)):
    document = re.sub(r'\W', ' ', str(data[sen]))
    print("Remove all the special characters                --> "+ document)

    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    print("remove all single characters                     --> "+ document)

    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
    print("Remove single characters from the start          --> "+ document)

    document = re.sub(r'\s+', ' ', document, flags=re.I)
    print("Substituting multiple spaces with single space   --> "+ document)

    document = re.sub(r'^b\s+', '', document)
    print("Removing prefixed 'b'                            --> "+ document)

    document = document.lower()
    print("Converting to Lowercase                          --> "+ document)

    document = document.split()
    print("Lemmatization                                    --> "+ document.__str__())

    document = [porterStemmer.stem(word+"\n")for word in document]

    document = ' '.join(document)
    print("Stemmer                                          --> "+ document.__str__())

    documents.append(document)
    mergedFile.write(documents[sen])

    print("End                                              --> " + documents[sen])
    print("***************************************************")