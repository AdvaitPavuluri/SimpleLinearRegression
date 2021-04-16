import pandas as pd
from sklearn.linear_model import LinearRegression
import argparse

class SimpleLR:
    def __init__(self, train_csv, test_csv):
        self.train_csv = train_csv
        self.test_csv = test_csv

    def test(self,model):
        self.dftest = pd.read_csv(self.test_csv)
        XTest = self.dftest['LotArea'].values.reshape((-1, 1))
        predicted = model.predict(XTest)
        return predicted
    def train(self):
        self.dftrain = pd.read_csv(self.train_csv)
        X = self.dftrain['LotArea'].values.reshape((-1, 1))
        Y = self.dftrain['SalePrice'].values
        model = LinearRegression()
        model.fit(X,Y)
        return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", required=True, type=str, help="training csv file")
    parser.add_argument("-p", required=True, type=str, help="prediction csv file")
    args = parser.parse_args()
    training_csv = args.t
    test_csv = args.p
    slr = SimpleLR(training_csv, test_csv)
    model = slr.train()
    results = slr.test(model)
    print(results)




