import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


data = pd.read_csv('updated_dataset.csv')
data = data.fillna("")
#train , test = train_test_split(data,test_size=0.20)

# print(train.shape)
# print(train.head())

# print(test.shape)
# print(test.head())


cv = CountVectorizer(token_pattern = r"(?u)\b\w+\b")

X_train = cv.fit_transform(data["content"])
encoded = cv.vocabulary_
encoded = {k:(v+3) for k,v in encoded.items()}
encoded["<PAD>"]= 0
encoded["<START>"]= 1
encoded["<UNK>"] = 2
encoded["<UNUSED>"] = 3
encoded[' ']= 30000


new_datacon=[]
for i in data['content']:
	k=[]
	for j in i:
		k.append(encoded[j])
	new_datacon.append[k]
	break
print(new_datacon)



#cv.vocabulary_.get(u'algorithm')
#print(cv.vocabulary_)