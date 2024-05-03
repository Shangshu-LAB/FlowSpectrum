

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

dataset=pd.read_csv('KDDtrain_.csv', index_col=0)
Labels=['Normal','Dos','Probe','R2L','U2R']
LabelMap=dict(zip(Labels,range(len(Labels))))


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from tensorflow.keras.utils import to_categorical
features=dataset.columns.drop(['class','label'])
num_classes=5
X,y = dataset[features],dataset['label'].apply(lambda x:LabelMap[x])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

preprocess=StandardScaler()
X_train_=preprocess.fit_transform(X_train)
X_test_=preprocess.transform(X_test)
y_train_, y_test_ = to_categorical(y_train,num_classes=num_classes),to_categorical(y_test,num_classes=num_classes)


from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Conv1D,MaxPooling1D,Input
from tensorflow.keras.layers import Flatten,Dense,Dropout
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE,CategoricalCrossentropy

activation='tanh'
input=Input(shape=(len(features),),name='input')
encoder=Dense(units=64,activation=activation)(input)
encoder=Dense(units=32,activation=activation)(encoder)
encoder=Dense(units=16,activation=activation)(encoder)
encoder=Dense(units=8,activation=activation)(encoder)

z1=Dense(units=1,name='z1')(encoder)
z2=Dense(units=1,name='z2')(encoder)
encoder=Concatenate(name='code')([z1,z2])

decoder=Dense(units=8,activation=activation)(encoder)
decoder=Dense(units=16,activation=activation)(decoder)
decoder=Dense(units=32,activation=activation)(decoder)
decoder=Dense(units=64,activation=activation)(decoder)
output=Dense(units=len(features),name='output')(decoder)

result=Dense(units=num_classes,activation='softmax',name='result')(z2)

AutoEncoder=Model(inputs=[input], outputs=[output], name='AutoEncoder')
Encoder=Model(inputs=[input], outputs=[encoder], name='Encoder')

model=Model(inputs=[input],outputs=[output,result],name='Model')
model.compile(
    optimizer='adam',
    loss={
        'output': 'mse',
        'result': 'categorical_crossentropy'
    },
    loss_weights={
        'output': 1.,
        'result': 1.
    },
    metrics={
        'result': 'accuracy'
    }
)


#%%
from tensorflow.keras.utils import plot_model
model.summary()
plot_model(model, to_file='KDD_model.png', show_shapes=True, )

from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint=ModelCheckpoint(
    filepath='KDD_model.h5',
    monitor='val_output_loss',verbose=1,
    save_best_only=True,mode='min'
)
history=model.fit(
    {'input':X_train_},{'output':X_train_,'result':y_train_},
    validation_data=({'input':X_test_},{'output':X_test_,'result':y_test_}),
    epochs=100, batch_size=256,
    callbacks=[checkpoint]
)


plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.plot(history.history['output_loss'],label='output_loss')
plt.plot(history.history['val_output_loss'],label='val_output_loss')
plt.plot(history.history['result_loss'],label='result_loss')
plt.plot(history.history['val_result_loss'],label='val_result_loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.plot(history.history['result_accuracy'],label='result_acc')
plt.plot(history.history['val_result_accuracy'],label='val_result_acc')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

