from tensorflow import keras 
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
 
# Initialising the CNN
classifier = Sequential()
 
# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, 
        input_shape = (64, 64, 3), activation = 'relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
 
# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
 
# Step 3 - Flattening
classifier.add(Flatten())
 
# Step 4 - Full connection
classifier.add(Dense(1, activation = 'relu'))
classifier.add(Dense(1, activation = 'sigmoid'))
 
# Compiling the CNN
classifier.compile(optimizer = 'adam', 
                   loss = 'binary_crossentropy',
                   metrics = ['accuracy'])

####FITTING IMAGE TO CNN#################################
from keras.preprocessing.image import ImageDataGenerator
 
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
 
test_datagen = ImageDataGenerator(rescale = 1./255)


training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
 
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = (1792/32),
                         epochs = 5,
                         validation_data = test_set,
                         validation_steps = (2000/32))


class_labels = {v: k for k, v in training_set.class_indices.items()}
print(class_labels)
######Predicting the Image############
from skimage.io import imread
from skimage.transform import resize
import numpy as np
 

img = imread('pexels-photo-257540.jpeg') 
img = resize(img,(64,64)) 
img = np.expand_dims(img,axis=0) 
#prediction = classifier.predict_classes(img)
prediction = (classifier.predict(img) > 0.5).astype("int32")
#print(prediction)
if prediction[0][0]==1:
    print("It is a dog")
elif prediction[0][0]==0:
    print("It is a cat")