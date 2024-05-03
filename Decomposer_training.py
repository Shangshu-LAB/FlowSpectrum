
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

dataset=pd.read_csv('KDDtrain.csv', index_col=0)
Labels=['Normal','Dos','Probe','R2L','U2R']
LabelMap=dict(zip(Labels,range(len(Labels))))


from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
features=dataset.columns.drop(['class','label'])
num_classes=5
X,y = dataset[features],dataset['label'].apply(lambda x:LabelMap[x])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train, X_test = X_train.astype('float64'), X_test.astype('float64')
y_train_, y_test_ = to_categorical(y_train,num_classes=num_classes),to_categorical(y_test,num_classes=num_classes)

from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
model=load_model('KDD_model.h5')
Encoder=Model(inputs=[model.input], outputs=[model.get_layer('code').output], name='Encoder')

from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint=ModelCheckpoint(
    filepath='KDD_model.h5',
    monitor='val_output_loss',verbose=1,
    save_best_only=True,mode='min'
)

model.compile(
    optimizer='adam',
    loss={'output': 'mse', 'result': 'categorical_crossentropy'},
    loss_weights={'output': 1.0,'result': 1.0},
    metrics={'result': 'accuracy'}
)
history=model.fit(
    {'input': X_train}, {'output': X_train, 'result': y_train_},
    validation_data=({'input': X_test}, {'output': X_test, 'result': y_test_}),
    epochs=50, batch_size=256,
    callbacks=[checkpoint]
)

pd.DataFrame(history.history).to_csv('history_.csv')

#
print(history.history.keys())
plt.plot(np.log10(history.history['loss']),label='loss')
plt.plot(np.log10(history.history['val_loss']),label='val_loss')
plt.plot(np.log10(history.history['output_loss']),label='output_loss')
plt.plot(np.log10(history.history['val_output_loss']),label='val_output_loss')
plt.plot(np.log10(history.history['result_loss']),label='result_loss')
plt.plot(np.log10(history.history['val_result_loss']),label='val_result_loss')
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


