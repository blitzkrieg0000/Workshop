from sklearn import  datasets, tree, metrics, model_selection

# LOAD DATASET
iris = datasets.load_iris()

# SPLIT - Test - Train
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.5)

# Set Algorithm
classifier = tree.DecisionTreeClassifier()

#!Train
classifier.fit(x_train, y_train)

#!Test
predictions = classifier.predict(x_test)

print(metrics.accuracy_score(y_test, predictions))