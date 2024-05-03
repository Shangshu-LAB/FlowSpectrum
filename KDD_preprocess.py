#%%

import pandas as pd
import numpy as np
from collections import Counter

features = [
    'duration',
    'protocol_type', 'service', 'flag',
    'src_bytes', 'dst_bytes', 'land',
    'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login',
    'count', 'srv_count',
    'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
    'class', 'hhh']

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


Trainset_path = 'Dataset/KDDTrain+.csv'
Testset_path='Dataset/KDDTest+.csv'
NSLKDD_train = pd.read_csv(Trainset_path , index_col=False, names=features)
NSLKDD_test = pd.read_csv(Testset_path , index_col=False, names=features)
NSLKDD_train['label']=NSLKDD_train['class'].apply(relabel)
NSLKDD_test['label']=NSLKDD_test['class'].apply(relabel)
# del NSLKDD_train['hhh'],NSLKDD_test['hhh'],NSLKDD_train['class'],NSLKDD_test['class'],
print('OK')

#%%

import matplotlib.pyplot as plt
import seaborn as sns
Other=[
    'wrong_fragment', 'urgent',
    'num_failed_logins', 'num_shells', 'num_access_files', 'num_outbound_cmds',
]
BoolFeatures = [
    'land', 'logged_in', 'is_host_login', 'is_guest_login', 'root_shell',
]
EnumFeatures = ['protocol_type', 'service', 'flag', ]
NumLogFeatures=[
    'duration','src_bytes','dst_bytes',
    'count', 'srv_count',
    'dst_host_count', 'dst_host_srv_count',
    'hot','num_compromised','num_root'
]
RateLogFeatures=[
    'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'num_file_creations'
]


SelectedFeatures=NumLogFeatures+RateLogFeatures+BoolFeatures+Other+EnumFeatures
X_train,X_test = NSLKDD_train[SelectedFeatures],NSLKDD_test[SelectedFeatures]
y_train,y_test = NSLKDD_train['label'],NSLKDD_test['label']

X_train=pd.get_dummies(X_train,columns=EnumFeatures)
X_test=pd.get_dummies(X_test,columns=EnumFeatures)

X_train['class']=NSLKDD_train['class']
X_test['class']=NSLKDD_test['class']
X_train['label']=NSLKDD_train['label']
X_test['label']=NSLKDD_test['label']

X_train.to_csv('KDDtrain_.csv')
X_test.to_csv('KDDtest_.csv')
