import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.core import  Lambda , Flatten

from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, Convolution2D , MaxPooling2D


base_dir = '../input/';

# load features, ignore header, ignore IDs
X_train1 = np.loadtxt(base_dir + 'train_features.csv', delimiter=',')[:,1:];
X_test1 = np.loadtxt(base_dir + 'test_features.csv', delimiter=',')[:,1:];
y_train1 = np.loadtxt(base_dir + 'train_labels.csv', dtype=np.uint8, delimiter=',', skiprows=1)[:,-1];

print(y_train1.shape[0])

#y = np_utils.to_categorical(y_train1)

im_train = X_train1[0,:].reshape((30,30,3), order='F')
im_test = X_test1[0,:].reshape((30,30,3), order='F')

plt.figure(1)
plt.imshow(im_train/255)
plt.axis('off')

plt.figure(2)
plt.imshow(im_test/255)
plt.axis('off');
#.....................................................
#train = pd.read_csv('../input/train.csv')
#labels = train.ix[:,0].values.astype('int32')
#X_train = (train.ix[:,1:].values).astype('float32')
#X_test = (pd.read_csv('../input/test.csv').values).astype('float32')

# convert list of labels to binary class matrix
#y_train = np_utils.to_categorical(y_train1) 

# pre-processing: divide by max and substract mean
#scale = np.max(X_train1)
#X_train1 /= scale
#X_test1 /= scale

#mean = np.std(X_train1)
#X_train1 -= mean
#X_test1 -= mean

mean_px = X_train1.mean().astype(np.float32)
std_px = X_train1.std().astype(np.float32)

def standardize(x):
    return (x-mean_px)/std_px


#transformer = RobustScaler().fit(x)  


input_dim = X_train1.shape[1]
print(X_train1.shape)
#nb_classes = y.shape[1]
X_train1 = X_train1.reshape(X_train1.shape[0],30,30,3)
X_test1 = X_test1.reshape(X_test1.shape[0],30,30,3)

X_train, X_test, y_train, y_test = train_test_split( X_train1, y_train1, test_size=0.1, random_state=1)
y_train1 = np_utils.to_categorical(y_train)
y_test1 = np_utils.to_categorical(y_test)

print(y_train1.shape)

from keras.preprocessing import image
gen = image.ImageDataGenerator()
batches = gen.flow(X_train, y_train1, batch_size=64)
val_batches=gen.flow(X_test, y_test1, batch_size=64)

#from sklearn import preprocessing
#pt = preprocessing.PowerTransformer(method='yeo-johnson', standardize=True)
#X_train = pt.fit_transform(X_train) 

# Read data

def get_bn_model():
    model = Sequential([
        Lambda(standardize, input_shape=(30,30,3)),
        Convolution2D(32,(3,3), activation='relu'),
        BatchNormalization(axis=1),
        Convolution2D(32,(3,3), activation='relu'),
        MaxPooling2D(),
        BatchNormalization(axis=1),
        Convolution2D(64,(3,3), activation='relu'),
        BatchNormalization(axis=1),
        Convolution2D(64,(3,3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        BatchNormalization(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dense(28, activation='softmax')
        ])
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
def get_bn_model2():
    model = Sequential([
        Lambda(standardize, input_shape=(30,30,3)),
        Convolution2D(32,(3,3), activation='relu'),
        BatchNormalization(axis=1),
        Convolution2D(32,(3,3), activation='relu'),
        MaxPooling2D(),
        BatchNormalization(axis=1),
        Convolution2D(64,(3,3), activation='relu'),
        BatchNormalization(axis=1),
        Convolution2D(64,(3,3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        BatchNormalization(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(28, activation='softmax')
        ])
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
from sklearn.metrics import accuracy_score
#model= get_bn_model()
#model.optimizer.lr=0.01
a1=0
b1=0
a2=0
b2=0
#for i in range(10): 
 #if(i<5):
model= get_bn_model2()
model.optimizer.lr=0.01
model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=1, 
                    validation_data=val_batches, validation_steps=val_batches.n)
 #else:
model2= get_bn_model2()
model2.optimizer.lr=0.01
model2.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=1, 
                    validation_data=val_batches, validation_steps=val_batches.n)

print("Generating test predictions...")
y_pred = model.predict_classes(X_train, verbose=0)
y_pred2 = model.predict_classes(X_test, verbose=0)
print("acc: ",accuracy_score(y_train, y_pred))
print("acc: ",accuracy_score(y_test, y_pred2))

y_pred = model2.predict_classes(X_train, verbose=0)
y_pred2 = model2.predict_classes(X_test, verbose=0)
print("acc: ",accuracy_score(y_train, y_pred))
print("acc: ",accuracy_score(y_test, y_pred2))
    
y_pred1a= model.predict(X_train, verbose=0)
y_pred1b=model2.predict(X_train, verbose=0)

y_pred2a = model.predict(X_test1, verbose=0)
y_pred2b = model2.predict(X_test1, verbose=0)
print('ok....ok')    
print(y_pred.shape,y_pred2b.shape)
y_pred5 = np.ones((y_pred2a.shape[0],y_pred2a.shape[1]))
from keras.utils import np_utils
print(y_pred2a.shape,y_pred2b.shape)
print(y_pred2a[:,0])
for i in range(y_pred5.size):
    #y_pred5[i]=np.argmax(( (y_pred2a))[i] )
    y_pred5[i]= (y_pred2a)[i] 

    
y_pred = np.ones(y_pred1a.shape[0])
y_pred2a=model.predict_classes(X_test1, verbose=0)
y_classes = keras.np_utils.probas_to_classes(y_pred5)
print(y_pred5==y_pred2a)
#print(y_pred2a.shape,y_pred2b.shape)
for i in range(y_pred.shape[0]):
    y_pred[i]=np.argmax(( (y_pred1a+y_pred1b)/2)[i] )    

print("acc444: ",accuracy_score(y_train, y_pred))
#print("acc555: ",accuracy_score(y_test, y_pred2))
print(y_test.shape,y_pred2.shape)

# y_pred = mode2l.predict_classes(X_train, verbose=0)
# y_pred2 = model2.predict_classes(X_test, verbose=0)

#print(preds)
# if(i<5):
 # a1+=accuracy_score(y_train, y_pred)
 # b1+=accuracy_score(y_test, y_pred2)
# else:
 # a2+=accuracy_score(y_train, y_pred)
 # b2+=accuracy_score(y_test, y_pred2)
  
#print(np.mean(y_pred==y_test))
# if(i==4):
#  print(a1/5)
#  print(b1/5)
# if(i==9):
#  print(a2/5)
 # print(b2/5)
df1 = pd.DataFrame(y_pred2a, columns=['Label'])
df1.index += 1 # upgrade to one-based indexing
df1.to_csv('submission1.csv',index_label='ID',columns=['Label'])
df = pd.DataFrame(y_pred5, columns=['Label'])
df.index += 1 # upgrade to one-based indexing
df.to_csv('submission2.csv',index_label='ID',columns=['Label'])
print(all(df==df1))
