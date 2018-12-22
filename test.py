import pickle

l = [1,2,3,4]
with open("/Users/keremdundar/Desktop/Kerem/Dersler/Bitirme_Projesi/Data/MergeByLabel/test.txt", "wb") as fp:   #Pickling
    pickle.dump(l, fp)

with open("/Users/keremdundar/Desktop/Kerem/Dersler/Bitirme_Projesi/Data/MergeByLabel/test.txt", "rb") as fp:   # Unpickling
    b = pickle.load(fp)
print(b)