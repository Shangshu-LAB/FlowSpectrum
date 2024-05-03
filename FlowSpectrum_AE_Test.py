
from sklearn.decomposition import PCA,TruncatedSVD,FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from datetime import datetime

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
        X_decompose=self.reconstructor.predict(X).reshape(-1)
        self.X_lim=[X_decompose.min(),X_decompose.max()]
        self.FlowSpectrum={}
        for label in Labels:
            self.FlowSpectrum[label]=X_decompose[y==label]

    def get_FlowSpectrum(self,title,save_path=None):
        # print(self.FlowSpectrum)
        plt.figure(figsize=(15,len(self.Labels)),dpi=400)
        for i,label in enumerate(self.Labels):
            plt.subplot(len(Labels),1,i+1)
            sns.rugplot(self.FlowSpectrum[label], height=1.,c='r')
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
        X_test_decompose=self.reconstructor.predict(X_test).reshape(-1)
        y_pred,y_true = [],[]
        times=[]
        for label in test_Label:
            data = X_test_decompose[y_test==label]
            for i in range(sample_count):
                sample = np.random.choice(a=data, size=sample_size)
                start=datetime.now()
                result = self.detect(sample=sample)
                end=datetime.now()
                y_pred.append(max(result, key=lambda k: result[k]))
                y_true.append(label)
                times.append((end-start).total_seconds())
        print('time:',np.mean(times))
        print(accuracy_score(y_pred=y_pred, y_true=y_true))
        print(classification_report(y_pred=y_pred, y_true=y_true,digits=4))
        if outputdict != None and name != None:
            pd.DataFrame(classification_report(
                y_pred=y_pred, y_true=y_true, digits=6, output_dict=True
            )).transpose().to_csv(f'{outputdict}/{name}_Report.csv')
            pd.DataFrame(confusion_matrix(
                y_pred=y_pred, y_true=y_true, labels=test_Label
            ), columns=test_Label, index=test_Label).to_csv(f'{outputdict}/{name}_Matrix.csv')


from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

Normal=[ 'normal',]
Dos=[ 'back', 'neptune', 'smurf', 'teardrop', 'land', 'pod',
      'apache2', 'mailbomb', 'processtable','udpstorm']
Probe=[ 'satan', 'portsweep', 'ipsweep', 'nmap',
        'mscan', 'saint',]
R2L=[ 'warezmaster', 'warezclient', 'ftp_write', 'guess_passwd', 'imap', 'multihop', 'phf', 'spy',
      'sendmail', 'named', 'snmpgetattack', 'snmpguess', 'xlock', 'xsnoop', 'worm',]
U2R=[ 'rootkit', 'buffer_overflow', 'loadmodule', 'perl',
      'httptunnel', 'ps', 'sqlattack', 'xterm',]
Labels=['Normal','Dos','Probe','R2L','U2R']
Classes=Normal+Dos+Probe+R2L+U2R
def relabel(classname):
    if classname in Normal:
        return 'Normal'
    elif classname in Dos:
        return 'Dos'
    elif classname in Probe:
        return 'Probe'
    elif classname in R2L:
        return 'R2L'
    elif classname in U2R:
        return 'U2R'
    return 'Unknown'

from sklearn.preprocessing import StandardScaler
import pickle
import joblib

if __name__=='__main__':
    ModelName = 'SemiAE'
    model = load_model('KDD_model.h5')
    Encoder = Model(inputs=[model.input], outputs=[model.get_layer('z2').output], name='Encoder')
    # model = load_model('KDD_model_no.h5')
    # Encoder = Model(inputs=[model.input], outputs=[model.get_layer('code').output], name='Encoder')

    preprocess = joblib.load('preprocess.joblib')

    trainset = pd.read_csv('KDDtrain_.csv', index_col=0)
    features = trainset.columns.drop(['class', 'label'])
    X, y = trainset[features], trainset['label']
    Labels = ['Normal', 'Dos', 'Probe', 'R2L', 'U2R']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    X_train_ = preprocess.transform(X_train)
    X_test_ = preprocess.transform(X_test)

    FS=FlowSpectrum(
        X=X_train_,y=y_train,
        Labels=Labels,
        reconstructor=Encoder,
    )
    FS.get_FlowSpectrum(title=f'FlowSpectrum Based on {ModelName}')


    testset = pd.read_csv('KDDtest_.csv', index_col=0)
    features = testset.columns.drop(['class', 'label'])
    X, y = testset[features], testset['label']
    X_ = preprocess.transform(X)
    FS.detection_test(
        X_test=X_, y_test=y,
        test_Label=Labels, sample_count=1000, sample_size=10,
        # outputdict='Result', name=ModelName
    )
