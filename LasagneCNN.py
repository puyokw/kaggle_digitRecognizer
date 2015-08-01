# 
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.layers import *
from lasagne.nonlinearities import softmax, rectify
from lasagne.updates import nesterov_momentum
from lasagne.updates import *
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import NeuralNet
import theano
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def load_train_data(path):
	df = pd.read_csv(path)
	X = df.values.copy()
	np.random.shuffle(X)
	X, labels = X[:, 1:].astype(np.float32), X[:, 0]
	encoder = LabelEncoder()
	y = encoder.fit_transform(labels).astype(np.int32)
	return X, y, encoder

def load_test_data(path):
    df = pd.read_csv(path)
    X = df.values.copy()
    X= X[:, 0:].astype(np.float32)
   # X = scaler.transform(X)
    return X

def float32(k):
    return np.cast['float32'](k)

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)


def make_submission(clf, X_test, encoder, name='my_neural_net_submission.csv'):
    y_prob = clf.predict_proba(X_test)
    with open(name, 'w') as f:
        f.write('0,1,2,3,4,5,6,7,8,9')
        #f.write(','.join(encoder.classes_))
        f.write('\n')
        for probs in y_prob:
            probas = ','.join( map(str,probs.tolist()))
            f.write(probas)
            f.write('\n')
    print("Wrote submission to file {}.".format(name))

np.random.seed(131)
X, y, encoder = load_train_data('C:\\Users\\kawa\\Desktop\\kaggle\\digitRecognizer\\train.csv')
print X
X_test = load_test_data('C:\\Users\\kawa\\Desktop\\kaggle\\digitRecognizer\\test.csv')
print X_test
num_classes = len(encoder.classes_)
num_features = X.shape[1]

#X=np.log(X+1)
X=np.sqrt(X+(3/8))
#X_test=np.log(X_test+1)
X_test=np.sqrt(X_test+(3/8))

X=X.reshape(-1,1,28, 28)
X_test=X_test.reshape(-1,1,28,28)

#for i in range(1,5):
net = NeuralNet(
    layers=[
        ('input', InputLayer),
        ('conv1', Conv2DLayer),
        ('pool1', MaxPool2DLayer),
        ('dropout1', DropoutLayer),
        ('conv2', Conv2DLayer),
        ('pool2', MaxPool2DLayer),
        ('dropout2', DropoutLayer),
        ('hidden3', DenseLayer),
        ('dropout3', DropoutLayer),
        ('hidden4', DenseLayer),
        ('dropout4', DropoutLayer),
        ('output', DenseLayer),
        ],
    input_shape=(None, 1, 28, 28),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    dropout1_p=0.2,
    conv2_num_filters=64, conv2_filter_size=(3, 3), pool2_pool_size=(2, 2),
    dropout2_p=0.2,
    hidden3_num_units=1024,
    dropout3_p=0.5,
    hidden4_num_units=1024,
    dropout4_p=0.5,
    output_num_units=num_classes,
    output_nonlinearity=softmax,
    update=adagrad,
    update_learning_rate=theano.shared(float32(0.03)),
    #update_momentum=theano.shared(float32(0.9)),
    eval_size=0.01,
    #regression=True,
    #batch_iterator_train=FlipBatchIterator(batch_size=128),
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.01, stop=0.0001),
        #AdjustVariable('update_momentum', start=0.9, stop=0.999),
        #EarlyStopping(patience=200),
        ],
    max_epochs=30,
    verbose=1,
    )

#clf3 = OneVsRestClassifier(SVC(C=5), n_jobs=-1)
#print "ES:"+X
#print y
net.fit(X, y)
nombre="C:\\Users\\kawa\\Desktop\\kaggle\\digitRecognizer\\LasagneCNNTest.csv"
make_submission(net, X_test, encoder,nombre)
#nombre="C:\\Users\\kawa\\Desktop\\kaggle\\digitRecognizer\\LasagneNNTrain.csv"
#make_submission(net, X, encoder,nombre)


