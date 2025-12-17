import numpy as np
from sklearn.linear_model import LogisticRegression

no_of_hours = np.array([2,5,1,8,9,3,4]).reshape(-1,1)      # 1 -> Number of coloumns needed
                                                           # -1 -> NUmpy should automatically compute number of rows needed

pass_fail = np.array([0,1,0,1,1,0,1])

model = LogisticRegression()
model.fit(no_of_hours,pass_fail)                           # here first a linear equation is created and linear result(z) is passed through sigmoid function

                                                           # Linear regression → gives any value from −∞ to +∞

                                                            #Logistic regression → converts that value into 0 to 1 probability

new_student = np.array([[3]])
print("Prediction",model.predict(new_student))
print('Prediction Probability',model.predict_proba(new_student))  #probability of predicting pass or fail