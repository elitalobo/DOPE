
import os
import sys
import numpy as np


class AnomalyDetector():
    def __init__(self, dirpath="data/", attack_type='influence',env='hiv', plotdir="anomaly_plots"):

        self.attack_type = attack_type
        self.dirpath = dirpath
        self.env = env
        self.plotdir = plotdir
        if  os.path.exists(self.plotdir):
            os.mkdir(self.plotdir)

    def isolation_forest(self, X, indices):
        from sklearn.ensemble import IsolationForest
        clf = IsolationForest(random_state=0)
        clf.fit(X)
        y = clf.predict(X)
        anm = np.zeros(X.shape[0])
        anm[indices]=-1
        anm1 = -1*np.ones(X.shape[0]) + anm*-1
        return np.array([np.sum(y==anm)/indices.shape[0], np.sum(y==anm1)/(X.shape[0]-indices.shape[0])])

    def local_outlier_factor(self,X, indices):
        import numpy as np
        from sklearn.neighbors import LocalOutlierFactor
        clf = LocalOutlierFactor()
        y = clf.fit_predict(X)
        anm = np.zeros(X.shape[0])
        anm[indices]=-1
        anm1 = -1*np.ones(X.shape[0]) + anm*-1
        return np.array([np.sum(y==anm)/indices.shape[0], np.sum(y==anm1)/(X.shape[0]-indices.shape[0])])

    def plot_all(self):
        res={}
        for file in os.listdir(self.dirpath):
            if self.attack_type in file and self.env in file:
                tokens = file.split("_")
                method = tokens[0]
                dataset = tokens[-1]
                is_type= tokens[-2]

                val = str(np.round(float(tokens[-5]),2))
                self.plot_name = method + "-" + self.attack_type + "-" + self.env + "-" + is_type + "-" + val
                path = self.dirpath + "/" + file + "/" + "corrupt_states.npy"
                path1 = self.dirpath + "/" + file + "/" + "influential_indices.npy"


                corrupt_states = np.load(path)
                indices = np.load(path1)
                outliers1 = self.isolation_forest(corrupt_states, indices)
                outliers2 = self.local_outlier_factor(corrupt_states, indices)

                # outliers3 = self.one_svm(corrupt_states, indices)
                # print("done 3")
                #
                # outliers4 = self.one_svm_sgd(corrupt_states, indices)
                # print("done 4")

                vals = np.array([outliers1,outliers2])
                if res.get(self.plot_name) is None:
                    res[self.plot_name]=[]
                res[self.plot_name].append(vals)

        for key, value in res.items():
            print(key)
            print(value)
            np.save(key+".npy", value)

anomaly = AnomalyDetector(env='custom')
anomaly.plot_all()

anomaly = AnomalyDetector(env='hiv')
anomaly.plot_all()

anomaly = AnomalyDetector(env='cancer')
anomaly.plot_all()








