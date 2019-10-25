# fine_tune_resnet50_keras

* problem statment was from hackerearth in which we had to Classify the Lunar Rock(achived 93% accuracy on test setd).

* Suggestion = 
  1 you should use dropout layer with dense layer in model to prevent it from overfitting.
  2 if you want to use other dataset then you just need to change the path and steps per epoch which is equal to  (total num of               images/batch size). 
  
APPROACH = 

* libraries used = keras, pandas, numpy.

* feature engineering = our dataset is completely new as compared to the imagenet dataset.
  
* Here we are using keras pretrained model resnet50 as our base model.

* we have imported the base model with pretrained weights which are achieved from training the model on imagenet dataset.
so we imported the model without the top layer because our dataset is new and the top layer identifies and relate
the features which are very specific to the problem domain although the lower layer identifies the general features 
like edges and shapes.

* we dont want to train the base model(resnet).so we have to set layer = false in base model.

* after this we made a sequential keras model in which we first add the base model then flatten layer which converts a 
matrix into a 1d array then after that we add dense layer of 1000 neuron after this we add a dense(output) layer 
of 2 neuron (because we have two class to differentiate) .

* we compiled the model using adam optimizer.

* we trained the model in 5 epoch and take steps_per_epoch = 1714 because we have 11998 total images and we took
batch size = 7.

* then we test the model and modify the test csv file according to its correct result given by the program.
