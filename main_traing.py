from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

tar=3
path='./Dataset'
# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    files = np.array(data['filenames'])
    targets = np_utils.to_categorical(np.array(data['target']), tar)
    return files, targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset(path)

test_files=train_files
test_targets = train_targets

# get the burn classes
# We only take the characters from a starting position to remove the path
#burn_classes = [item[11:-1] for item in sorted(glob(path))]
burn_classes = [item[7:-1] for item in sorted(glob("./Dataset/*/"))]
print(burn_classes)
# print statistics about the dataset
print('There are %d total categories.' % len(burn_classes))
print(burn_classes)
print('There are %s total burn images.\n' % len(np.hstack([train_files, test_files])))
print('There are %d training images.' % len(train_files))
print('There are %d test images.'% len(test_files))

for file in train_files: assert('.DS_Store' not in file)



from keras.preprocessing import image                  
from tqdm import tqdm

# Note: modified these two functions, so that we can later also read the inception tensors which 
# have a different format 
def path_to_tensor(img_path, width=224, height=224):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(width, height))
    # convert PIL.Image.Image type to 3D tensor with shape (width, heigth, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, width, height, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths, width=224, height=224):
    list_of_tensors = [path_to_tensor(img_path, width, height) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

import keras
import timeit

# graph the history of model.fit
def show_history_graph(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show() 


def show_history_graph1(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show() 
# callback to show the total time taken during training and for each epoch
class EpochTimer(keras.callbacks.Callback):
    train_start = 0
    train_end = 0
    epoch_start = 0
    epoch_end = 0
    
    def get_time(self):
        return timeit.default_timer()

    def on_train_begin(self, logs={}):
        self.train_start = self.get_time()
 
    def on_train_end(self, logs={}):
        self.train_end = self.get_time()
        print('Training took {} seconds'.format(self.train_end - self.train_start))
 
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_start = self.get_time()
 
    def on_epoch_end(self, epoch, logs={}):
        self.epoch_end = self.get_time()
        print('Epoch {} took {} seconds'.format(epoch, self.epoch_end - self.epoch_start))

from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255


from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.models import Model
from keras.callbacks import ModelCheckpoint 

import matplotlib.pyplot as plt





img_width, img_height = 224, 224
batch_size = 8
num_classes = 4
epoch=10



from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten()) 
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(tar, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
hist=model.fit(train_tensors, train_targets ,validation_split=0.1, epochs=epoch, batch_size=64)

show_history_graph(hist)
test_loss, test_acc = model.evaluate(train_tensors, train_targets)

y_pred=model.predict(test_tensors)
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(np.argmax(test_targets, axis=1),np.argmax(y_pred, axis=1))

accuracyrcnn = accuracy_score(np.argmax(test_targets, axis=1),np.argmax(y_pred, axis=1))
print("Incepption ResNetV2 confusion matrics=",cm)
print("  ")
print("Incepption ResNetV2 accuracy=",accuracyrcnn*100)

from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
f1_score=f1_score(np.argmax(test_targets, axis=1),np.argmax(y_pred, axis=1), average='weighted')  
precision, recall, thresholds = precision_recall_curve( np.argmax(test_targets, axis=1)>1,np.argmax(y_pred, axis=1)>1)
print(' Incepption ResNetV2 Precision ='+str(np.mean(precision)*100))
print('Incepption ResNetV2 Recall ='+str(np.mean(recall)*100))
print('Incepption ResNetV2 F1 score ='+str(np.mean(f1_score)*100))


from sklearn.metrics import roc_curve
# Calculate ROC curve from y_test and pred
fpr, tpr, thresholds = roc_curve(np.argmax(test_targets, axis=1)>=1,np.argmax(y_pred, axis=1)>=1)



# Plot the ROC curve
fig = plt.figure(figsize=(8,8))
plt.title('Receiver Operating Characteristic')

# Plot ROC curve
plt.plot(fpr, tpr, label='l1')
plt.legend(loc='lower right')

# Diagonal 45 degree line
plt.plot([0,1],[0,1],'k--')

# Axes limits and labels
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('Incepption ResNetV2 True Positive Rate')
plt.xlabel('Incepption ResNetV2 False Positive Rate')
plt.show()




model.save('trained_model_Incepption_ResNetV2.h5')







########

img_width, img_height = img_width, img_height
batch_size = 32
samples_per_epoch = 10
validation_steps = 300
nb_filters1 = 32
nb_filters2 = 64
conv1_size = 3
conv2_size = 3
pool_size = 3
classes_num = 4
lr = 0.0004
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import callbacks
import time
#input_shape=(img_width, img_height,3)
model = Sequential()
model.add(Convolution2D(nb_filters1, conv1_size, conv1_size, border_mode ="same", input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Convolution2D(nb_filters2, conv2_size, conv2_size, border_mode ="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(tar, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=lr),
              metrics=['accuracy'])

hist=model.fit(train_tensors, train_targets ,validation_split=0.1, epochs=epoch, batch_size=64)

show_history_graph(hist)
test_loss, test_acc = model.evaluate(train_tensors, train_targets)

y_pred=model.predict(test_tensors)
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(np.argmax(test_targets, axis=1),np.argmax(y_pred, axis=1))

accuracydnn = accuracy_score(np.argmax(test_targets, axis=1),np.argmax(y_pred, axis=1))
print("DenseNet121 confusion matrics=",cm)
print("  ")
print("DenseNet121 accuracy=",accuracydnn*100)

from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
f1_score=f1_score(np.argmax(test_targets, axis=1),np.argmax(y_pred, axis=1), average='weighted')  
precision, recall, thresholds = precision_recall_curve( np.argmax(test_targets, axis=1)>1,np.argmax(y_pred, axis=1)>1)
print('DenseNet121 Precision ='+str(np.mean(precision)*100))
print('DenseNet121 Recall ='+str(np.mean(recall)*100))
print('DenseNet121 F1 score ='+str(np.mean(f1_score)*100))


from sklearn.metrics import roc_curve
# Calculate ROC curve from y_test and pred
fpr, tpr, thresholds = roc_curve(np.argmax(test_targets, axis=1)>=1,np.argmax(y_pred, axis=1)>=1)



# Plot the ROC curve
fig = plt.figure(figsize=(8,8))
plt.title('Receiver Operating Characteristic')

# Plot ROC curve
plt.plot(fpr, tpr, label='l1')
plt.legend(loc='lower right')

# Diagonal 45 degree line
plt.plot([0,1],[0,1],'k--')

# Axes limits and labels
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('DenseNet121 True Positive Rate')
plt.xlabel('DenseNet121 False Positive Rate')
plt.show()




model.save('trained_model_DenseNet121.h5')

#############


import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

input_shape=(img_width, img_height,3)
model = tf.keras.models.Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Conv2D(128, (3, 3), activation='relu'),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Conv2D(256, (3, 3), activation='relu'),
    Conv2D(256, (3, 3), activation='relu'),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dense(512, activation='relu'),
    Dense(tar, activation='softmax')
])
# In[11]:
model.summary()
# In[12]:

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics =['accuracy'])

hist=model.fit(train_tensors, train_targets ,validation_split=0.1, epochs=epoch, batch_size=64)

show_history_graph1(hist)
test_loss, test_acc = model.evaluate(train_tensors, train_targets)

y_pred=model.predict(test_tensors)
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(np.argmax(test_targets, axis=1),np.argmax(y_pred, axis=1))

accuracycnn = accuracy_score(np.argmax(test_targets, axis=1),np.argmax(y_pred, axis=1))
print("Ensemble method confusion matrics=",cm)
print("  ")
print("Ensemble method accuracy=",accuracycnn*100)


from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
f1_score=f1_score(np.argmax(test_targets, axis=1),np.argmax(y_pred, axis=1), average='weighted')  
precision, recall, thresholds = precision_recall_curve( np.argmax(test_targets, axis=1)>1,np.argmax(y_pred, axis=1)>1)
print('Ensemble method Precision ='+str(np.mean(precision)*100))
print('Ensemble method Recall ='+str(np.mean(recall)*100))
print('Ensemble method F1 score ='+str(np.mean(f1_score)*100))

from sklearn.metrics import roc_curve
# Calculate ROC curve from y_test and pred
fpr, tpr, thresholds = roc_curve(np.argmax(test_targets, axis=1)>=1,np.argmax(y_pred, axis=1)>=1)



# Plot the ROC curve
fig = plt.figure(figsize=(8,8))
plt.title('Receiver Operating Characteristic')

# Plot ROC curve
plt.plot(fpr, tpr, label='l1')
plt.legend(loc='lower right')

# Diagonal 45 degree line
plt.plot([0,1],[0,1],'k--')

# Axes limits and labels
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('Ensemble method True Positive Rate')
plt.xlabel('Ensemble method False Positive Rate')
plt.show()
model.save('trained_model_Ensemble_method.h5')

#

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = ['RCNN', 'DNN', 'CNN']
students = [accuracyrcnn,accuracydnn,accuracycnn]
ax.bar(langs,students)
plt.show()


