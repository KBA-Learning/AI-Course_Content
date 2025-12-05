import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Generating Random Dataset
np.random.seed(42)     #setting seed for random number as 42 inorder to get the same series
x=np.random.rand(50,1)*100
print(x)
Y = 3.5 * x+ np.random.randn(50, 1) * 20 
print('Y',Y)

#Training the model
model = LinearRegression()      #creating an object on LinearRegression
model.fit(x,Y)

#Predicting Y based on x
y_pred = model.predict(x)
print('y predict',y_pred)

plt.figure(figsize=(8,6)) 
plt.scatter(x, Y, color='blue', label='Data Points') 
plt.plot(x, y_pred, color='red', linewidth=2, label='Regression Line') 
plt.title('Linear Regression on Random Dataset')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.savefig("/home/lekshmipg/AI/DataEncoding/dencode/plot.png")
plt.close()

