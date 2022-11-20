import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier



class ClassifierKNN(object):

    def __init__(self):
        self.features = None
        self.featureNames = None
        self.labels = None
        self.labelNames = None
        self.classifier = None


    def GetExampleData(self):
        dataset = load_iris()
        return dataset.data, dataset.feature_names, dataset.target, list(dataset.target_names)


    def SplitData(self, features, labels, test_size=0.33, random_state=42):
        return train_test_split(features, labels, test_size=test_size, random_state=random_state)


    def SummaryData(self, featuresDF:pd.DataFrame):
        if isinstance(featuresDF, pd.DataFrame):
            print(featuresDF.describe(), "\n")
            print(featuresDF.info(), "\n")


    def DrawGraph(self, featuresDF, **kwargs):
        featuresDF.plot(**kwargs) # kind: "bar","box","scatter"


    def Convert2DataFrame(self, features, featureNames):
        if isinstance(features, np.ndarray):
            featuresDF = pd.DataFrame(features)
            featuresDF.columns = featureNames
            return featuresDF


    def PrepareModel(self, **kwargs):
        self.classifier = KNeighborsClassifier(**kwargs)


    def Train(self, xTrain, yTrain):
        self.classifier.fit(xTrain, yTrain)


    def Test(self, xTest, yTest):
        return self.classifier.score(xTest, yTest)


    def SaveModel(self, path="weights/knn_model.pickle"):
        with open(path, "wb") as model_file:
            pickle.dump(self.classifier, model_file, protocol=pickle.HIGHEST_PROTOCOL)


    def LoadModel(self, path="weights/knn_model.pickle"):
        with open(path, "rb") as model_file:
            self.classifier = pickle.load(model_file)



if '__main__' == __name__:
    classifier = ClassifierKNN()


    #! Veriyi yükle, "Özellik:Etiket"
    features, featureNames, labels, labelNames = classifier.GetExampleData()


    #! Veriyi görselleştir
    dataFrame:pd.DataFrame = classifier.Convert2DataFrame(features, featureNames)
    classifier.SummaryData(dataFrame)
    classifier.DrawGraph(dataFrame, x="sepal length (cm)", y="sepal width (cm)", kind="scatter")


    #! Veriyi Train:Test ayır
    xTrain, xTest, yTrain, yTest = classifier.SplitData(features, labels)


    #! Modeli hazırla, hiperparametreleri gir
    classifier.PrepareModel(n_neighbors=8)


    #! Modeli Eğit
    classifier.Train(xTrain, yTrain)


    #! Modeli hiç görmediği verilerle test et
    accuracy = classifier.Test(xTest, yTest)
    print(f"Test datası ile alınan gerçek score: {accuracy}")


    #! Modeli bildiği verilerle test et
    accuracy = classifier.Test(xTrain, yTrain)
    print(f"Train datası ile alınan ezber score: {accuracy}")


    #TODO Model Underfit mi? Overfit mi oldu? Accuracyleri karşılaştır.