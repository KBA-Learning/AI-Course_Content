from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

data = ['Red','Green','Blue','Yellow','Green','Red']

#Label Encoding
le = LabelEncoder()
encoded_data = le.fit_transform(data)
print('Encoded Data',encoded_data)

#What LabelEncoder do internally
le1 = LabelEncoder()
encoder = le1.fit(data)
linear = encoder.transform(data)
print(linear)

#One Hot
df = pd.DataFrame(data,columns=['Colour'])         #convert to 2D array, here we need data in a 2D array format that is why DataFrame is used
one_hot = pd.get_dummies(df['Colour'],dtype=int)   #dtype=int ,convert bool to integer
print("One Hot Encoding")
print(one_hot)

#Ordinal Encoding
data1 = [['Low'],['High'],['Low'],['Medium'],['High'],['Low']]
oe = OrdinalEncoder(categories=[['Low','Medium','High']])
ordinal = oe.fit_transform(data1)
print("Ordinal Encoding")
print(ordinal)

#OneHot using sklearn
df = pd.DataFrame(data,columns=['Color'])
oh = OneHotEncoder(sparse_output=False,dtype=int)
onehot = oh.fit_transform(df[['Color']])
feature_names = oh.get_feature_names_out(['Color'])
one_hot_df = pd.DataFrame(onehot, columns=feature_names)
print("One Hot Encoding")
print(onehot)
print(one_hot_df)

