from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils
from keras.optimizers import Adam, SGD, Optimizer
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.ensemble import BaggingClassifier 
from sklearn.cross_validation import StratifiedKFold, KFold
path = './'


batch_size = 256
nb_classes = 10

img_rows, img_cols = 28, 28 # input image dimensions
nb_filters = 32 # number of convolutional filters to use
nb_pool = 2 # 2 size of pooling area for max pooling
nb_conv = 3 # 3 convolution kernel size

# the data, shuffled and split between tran and test sets

train = pd.read_csv(path+'train.csv')
labels = train['label']
del train['label']

test = pd.read_csv(path+'test.csv')

train = train.values
train = train.reshape(train.shape[0], 1, img_rows, img_cols)
test = test.values
test = test.reshape(test.shape[0], 1, img_rows, img_cols)
train = train.astype("float32")
test = test.astype("float32")
train /= 255
test /= 255
print('train shape:', train.shape)
print(train.shape[0], 'train samples')
print(test.shape[0], 'test samples')
label = np_utils.to_categorical(labels, nb_classes)

# convert class vectors to binary class matrices

N = train.shape[0]
trainId = np.array(range(N))
submissionTr = pd.DataFrame(index=trainId,columns=np.array(range(10)))
nfold=5 
RND = np.random.randint(0,10000,nfold)
pred = np.zeros((test.shape[0],10))
score = np.zeros(nfold)
i=0
skf = StratifiedKFold(labels, nfold, random_state=1337)
for tr, te in skf:
	X_train, X_valid, y_train, y_valid = train[tr], train[te], label[tr], label[te]
	predTr = np.zeros((X_valid.shape[0],10))
	n_bag=5   
	for j in range(n_bag):
		print('nfold: ',i,'/',nfold, ' n_bag: ',j,' /',n_bag)
		print("Building model...")
		model = Sequential()
		model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
			border_mode='full',
			input_shape=(1, img_rows, img_cols)))
		model.add(Activation('relu'))
		model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
		model.add(Dropout(0.25))
		model.add(Activation('relu'))
		model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
		model.add(Dropout(0.25))
		model.add(Flatten())
		model.add(Dense(128))
		model.add(Activation('relu'))
		model.add(Dropout(0.25))		
		model.add(Dense(nb_classes))
		model.add(Activation('softmax'))
		earlystopping=EarlyStopping(monitor='val_loss', patience=10, verbose=1)
		checkpointer = ModelCheckpoint(filepath=path+"weights.hdf5", verbose=0, save_best_only=True)
		model.compile(loss='categorical_crossentropy', optimizer='adadelta')
		model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1000, show_accuracy=True, 
		verbose=2, validation_data=(X_valid,y_valid), callbacks=[earlystopping,checkpointer])
		model.load_weights(path+"weights.hdf5")
		print("Generating submission...")
		pred += model.predict_proba(test)
		predTr += model.predict_proba(X_valid)
	predTr /= n_bag
	submissionTr.iloc[te] = predTr
	score[i]= log_loss(y_valid,predTr,eps=1e-15, normalize=True)
	print(score[i])
	i+=1

pred /= (nfold * n_bag)
print("ave: "+ str(np.average(score)) + "stddev: " + str(np.std(score)))
print(confusion_matrix(labels, submissionTr.idxmax(axis=1)))

pd.DataFrame(pred).to_csv(path+"kerasCNN.csv",index_label='ImageId')
Label=pd.DataFrame(pred).idxmax(axis=1)
submission = pd.DataFrame({'ImageId': np.array(range(test.shape[0]))+1, 'Label': Label})
submission.to_csv(path+"kerasCNN_submission.csv",index=False)

print(log_loss(labels,submissionTr.values,eps=1e-15, normalize=True))
submissionTr.to_csv(path+"kerasCNN_stack.csv",index_label='ImageId')

# nfold 5, bagging 5: 0.020957301 + 0.00140977765 , Public LB: 0.99371
# batch_size 256: 0.0203983009777 + 0.00172547876286, Public LB: 0.99414
