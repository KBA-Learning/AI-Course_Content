from sklearn.tree import DecisionTreeClassifier,plot_tree
import matplotlib.pyplot as plt
import numpy as np

x=np.array([[2],[4],[5],[8]]).reshape(-1,1)

y=np.array([0,0,1,1])

model = DecisionTreeClassifier()
model.fit(x,y)

print('Model Predict',model.predict([[6]]))

plt.figure(figsize=(10,6))
plot_tree(model,filled=True)
plt.savefig('/home/lekshmipg/AI/DataEncoding/dencode/decision.png')
plt.close()