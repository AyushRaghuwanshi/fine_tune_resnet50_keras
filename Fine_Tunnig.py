#!/usr/bin/env python
# coding: utf-8

# # Importing libraries which we are going to use

# In[ ]:


import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
import pandas as pd
from keras.models import load_model
from keras.preprocessing import image


# # Defining the train images path and partial test images path in which we have to add the image to convert it into full path.

# In[ ]:


train_path = 'DataSet/Train Images'
test_path = 'DataSet\Test Images\\'


# In[ ]:


train_batches = ImageDataGenerator().flow_from_directory(train_path,target_size = (224,224),classes=['Large','Small'],batch_size = 7)


# # Importing keras pretrained model (without top layers) resnet50 which we are going to use as base model.

# In[ ]:


base_model = keras.applications.resnet50.ResNet50( weights = 'imagenet',include_top = False,input_shape = (224,224,3))


# # we dont want to train base model.

# In[ ]:


for layer in base_model.layers:
    layer.trainable = False


# # Making keras sequential model by using base model layer as its lower layer and two fully connected dense layer as its top layer which are trainable.

# In[ ]:


model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(1000,activation = 'relu'))
model.add(Dense(2,activation = 'sigmoid'))
print(model.summary())


# # Compile and train the model on epoch = 5 

# In[ ]:


model.compile(optimizer=Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[ ]:


history = model.fit_generator(train_batches,
                              epochs=5,
                              steps_per_epoch = 1714,
                             )


# # saving the model for future use

# In[ ]:


model.save('lunar.h5')
print("lunar.h5 was saved")


# # Reading the csv file as dataframe to get the name of images 

# In[ ]:


names = pd.read_csv('DataSet/test.csv',dtype = {'Image_file':str, 'Class':str})


# # Defining a function predict which takes the full path of image as argument and return the predicted class label

# In[ ]:


def predict(img_path):
    test_image = image.load_img(img_path, target_size=(224, 224))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image, batch_size=1)
    print(result)
    if(result[0][1]==0):
        return "Large"
    else:
        return "Small"

    
    


# # we are changing the class value (in front of its image name) according to the predicted value 

# In[ ]:



for i in range(len(names)):
    image_path = test_path + names.iloc[i][0]
    label = predict(image_path)
    names.at[i,'Class'] = label


# # saving modified dataframe in csv format.

# In[ ]:


names.to_csv("result.csv",index = False)


# In[ ]:




