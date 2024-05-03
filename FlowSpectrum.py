
from sklearn.decomposition import PCA,TruncatedSVD,FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def mul(Spectrum, sample):
    similar=[]
    for value in sample:
        idx=np.abs(Spectrum - np.asarray(value)).argmin()
        similar.append(Spectrum[idx])
    P=np.exp(-np.abs(np.asarray(sample)-np.asarray(similar))).mean()
    return P
def softmax(data):
    exp=np.exp(np.asarray(list(data.values())))
    return dict(zip(data.keys(),exp/exp.sum()))

class FlowSpectrum:
    def __init__(self,X,y,Labels,reconstructor):
        self.Labels=Labels
        self.reconstructor=reconstructor
        X_decompose=self.reconstructor.transform(X).reshape(-1)
        self.X_lim=[X_decompose.min(),X_decompose.max()]
        self.FlowSpectrum={}
        for label in Labels:
            self.FlowSpectrum[label]=X_decompose[y==label]

    def get_FlowSpectrum(self,title,save_path=None):
        # print(self.FlowSpectrum)
        plt.figure(figsize=(15,len(self.Labels)),dpi=800)
        for i,label in enumerate(self.Labels):
            plt.subplot(len(Labels),1,i+1)
            sns.rugplot(self.FlowSpectrum[label], height=1.)
            plt.xlim(self.X_lim)
            plt.title(label)
        plt.suptitle(title)
        plt.tight_layout()
        if save_path==None:
            plt.show()
        else:
            plt.savefig(save_path)

    def detect(self,sample):
        result={}
        for label in self.FlowSpectrum.keys():
            result[label] = mul(Spectrum=self.FlowSpectrum[label], sample=sample)
        # 可以增加softmax层
        return softmax(result)

    def detection_test(self,X_test,y_test,test_Label,outputdict=None,name=None,sample_count=100,sample_size=10):
        X_test_decompose=self.reconstructor.transform(X_test).reshape(-1)
        y_pred,y_true = [],[]
        for label in test_Label:
            data = X_test_decompose[y_test==label]
            for i in range(sample_count):
                sample = np.random.choice(a=data, size=sample_size)
                result = self.detect(sample=sample)
                y_pred.append(max(result, key=lambda k: result[k]))
                y_true.append(label)
        print(accuracy_score(y_pred=y_pred, y_true=y_true))
        print(classification_report(y_pred=y_pred, y_true=y_true))
        if outputdict!=None and name!=None:
            pd.DataFrame(classification_report(
                y_pred=y_pred, y_true=y_true,digits=6,output_dict=True
            )).transpose().to_csv(f'{outputdict}/{name}_Report.csv')
            pd.DataFrame(confusion_matrix(
                y_pred=y_pred, y_true=y_true,labels=test_Label
            ),columns=test_Label,index=test_Label).to_csv(f'{outputdict}/{name}_Matrix.csv')

