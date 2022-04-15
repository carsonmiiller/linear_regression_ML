import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

# use pandas to read in our csv file to variable data
# data = pd.read_csv("student-mat.csv", sep=";")
data = pd.read_csv("NBA_2022_advancedStats.csv")

# this selects only certain columns of data to remain in our data variable
# we only want variables we will use as "attributes" and "labels"
# data = data[["G1", "G2", "G3",  "studytime", "absences", "failures"]]
data = data[["MOV", "ORtg", "DRtg", "eFG%", "TOV%", "ORB%", "FT/FGA", "eFG%", "TOV%", "DRB%", "FT/FGA"]]


# choosing which of our variables is the one we will want to predict
predict = "MOV"
labels = np.array(data[predict])
################ Train NN with past games as input, and MOV/ATSW as output, test it on upcoming games, then use them as training material

# create an array from data, dropping the the predict variable, as we only want our attributes as independent variables
attributes = np.array(data.drop(predict, axis=1))
# create an array from data, keeping only the predict variable, as this will be our labels array

max_acc = .97


for x in range(2500):
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
    max_acc = max(max_acc, acc)
    if(max_acc == acc):
        print("Maximum Accuracy: \n", max_acc)
        # display coefficients of attributes (weighted importance)
        x = linear.coef_
        print("Coefficient: \n", linear.coef_)
        # display intercept of whole lin_reg eq
        print("Intercept: \n", linear.intercept_)

        # basically tells the lin_reg model to predict outputs for test inputs
        predictions = linear.predict(attributes_test)

        # displays predicted outputs, while showing the provided test inputs and also correct test outputs
        for i in range(len(predictions)):
            print("\nPredicted Avg MOV: ", predictions[i])
            print("Attributes: ", attributes_test[i])
            print("Actual Avg MOV: ", labels_test[i])