import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split 
from keras.utils import to_categorical 
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
data = []
labels = []
classes = 43
cur_path = os.path.dirname(os.path.abspath(__file__))
# print(cur_path)

# Retrieving the images and their labels
for i in range(classes):
   path = os.path.join(cur_path,'Train',str(i))
   images = os.listdir(path)
   for a in images:
        try:
           image = Image.open(path + '\\'+ a)
           image = image.resize((30,30))
           image = np.array(image)
          #sim = Image.fromarray(image)
           data.append(image)
           labels.append(i)
        except:
           print("Error loading image")
           
classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }
      
# # Iterating over the test folders    
# folders = os.listdir(cur_path + '\\Train')
# train_number = []
# class_num = []
# for folder in folders:
#     train_files = os.listdir(cur_path + '\\Train\\' + folder)
#     train_number.append(len(train_files))
#     class_num.append(classes[int(folder)])
    
# # Sorting the dataset on the basis of number of images in each class
# zipped_lists = zip(train_number, class_num)
# sorted_pairs = sorted(zipped_lists)
# tuples = zip(*sorted_pairs)
# train_number, class_num = [ list(tuple) for tuple in  tuples]

# # Plotting the number of images in each class
# plt.figure(figsize=(21,10))  
# plt.bar(class_num, train_number)
# plt.xticks(class_num, rotation='vertical')
# plt.show() 

# Visualizing 25 random images from test data
import random
from matplotlib.image import imread

test = pd.read_csv(cur_path + '\\Test.csv')
imgs = test["Path"].values

plt.figure(figsize=(25,25))

for i in range(1,26):
    plt.subplot(5,5,i)
    random_img_path = cur_path + '\\' + random.choice(imgs)
    rand_img = imread(random_img_path)
    plt.imshow(rand_img)
    plt.grid(b=None)
    plt.xlabel(rand_img.shape[1], fontsize = 20)#width of image
    plt.ylabel(rand_img.shape[0], fontsize = 20)#height of image
plt.show()

# Converting lists into numpy arrays
data = np.array(data)
labels = np.array(labels)
print(data.shape, labels.shape)

# Splitting training and testing dataset
X_t1, X_t2, y_t1, y_t2 = train_test_split(data, labels, test_size=0.2, random_state=42)
print(X_t1.shape, X_t2.shape, y_t1.shape, y_t2.shape)

# Converting the labels into one hot encoding
y_t1 = to_categorical(y_t1, 43)
y_t2 = to_categorical(y_t2, 43)

# Building the model
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5,5), activation='relu', input_shape = X_t1.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

# Compilation of the model
eps = 50
lr = 1e-5
opt = tf.keras.optimizers.Nadam(learning_rate = lr)
model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
anc = model.fit(X_t1, y_t1, batch_size=256, epochs=eps, validation_data=(X_t2, y_t2))
mn = "1f"
model.save("my_model_" + mn + ".h5")

# Plotting graphs for accuracy
plt.figure(0)
plt.plot(anc.history['accuracy'], label='training accuracy')
plt.plot(anc.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
plt.figure(1)
plt.plot(anc.history['loss'], label='training loss')
plt.plot(anc.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

# Testing accuracy on validation dataset
from sklearn.metrics import accuracy_score
y_test = pd.read_csv(cur_path + "\\Test.csv")
labels = y_test["ClassId"].values
imgs = y_test["Path"].values
data=[]
for img in imgs:
   image = Image.open(cur_path + "\\" + img)
   image = image.resize((30,30))
   data.append(np.array(image))
X_test=np.array(data)
pred = model.predict(X_test)

# Accuracy with the test data
from sklearn.metrics import accuracy_score
print(accuracy_score(labels, pred.argmax(axis=1)))
model.save("traffic_classifier_" + mn + ".h5")
