import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

# use pandas to read in our csv file to variable data
data = pd.read_csv("student-mat.csv", sep=";")

# this selects only certain columns of data to remain in our data variable
# we only want variables we will use as "attributes" and "labels"
data = data[["G1", "G2", "G3", "studytime", "absences", "failures"]]

# choosing which of our variables is the one we will want to predict
predict = "G3"

# create an array from data, dropping the the predict variable, as we only want our attributes as independent variables
attributes = np.array(data.drop([predict], 1))
# create an array from data, keeping only the predict variable, as this will be our labels array
labels = np.array(data[predict])

# split up these two arrays into 4:
# - 1 for training inputs
# - 1 for training outputs
# - 1 for testing inputs
# - 1 for testing outputs
attributes_train, attributes_test, labels_train, labels_test = sklearn.model_selection.train_test_split(attributes, labels, test_size = 0.1)

# initialize linear as a linear regression model
linear = linear_model.LinearRegression()

# create a best fit line from the training inputs and training outputs
linear.fit(attributes_train, labels_train)

# see how accurately this best fit line fits the test inputs and outputs
acc = linear.score(attributes_test, labels_test)
print("Accuracy: \n", acc)

# display coefficients of attributes (weighted importance)
print("Coefficient: \n", linear.coef_)
# display intercept of whole lin_reg eq
print("Intercept: \n", linear.intercept_)

# basically tells the lin_reg model to predict outputs for test inputs
predictions = linear.predict(attributes_test)

# displays predicted outputs, while showing the provided test inputs and also correct test outputs
for i in range(len(predictions)):
    print("\nPredicted Final Grade: ", predictions[i])
    print("Attributes: ", attributes_test[i])
    print("Actual Final Grade: ", labels_test[i])