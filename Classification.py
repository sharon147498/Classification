import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
from tensorflow import keras


image_dir=r"C:\Users\sharo\OneDrive\Desktop\Practical Assignmnt - DL\Dataset_Celebrities\cropped"
lionel_messi=os.listdir(r"C:\Users\sharo\OneDrive\Desktop\Practical Assignmnt - DL\Dataset_Celebrities\cropped\lionel_messi")
maria_sharapova=os.listdir(r"C:\Users\sharo\OneDrive\Desktop\Practical Assignmnt - DL\Dataset_Celebrities\cropped\maria_sharapova")
roger__federer=os.listdir(r"C:\Users\sharo\OneDrive\Desktop\Practical Assignmnt - DL\Dataset_Celebrities\cropped\roger_federer")
serena_williams=os.listdir(r"C:\Users\sharo\OneDrive\Desktop\Practical Assignmnt - DL\Dataset_Celebrities\cropped\serena_williams")
virat_kohli=os.listdir(r"C:\Users\sharo\OneDrive\Desktop\Practical Assignmnt - DL\Dataset_Celebrities\cropped\virat_kohli")


print("--------------------------------------\n")

print('The length of lionel_messi images is',len(lionel_messi))
print('The length of aria_sharapova images is',len(maria_sharapova))
print('The length of roger__federer images is',len(roger__federer))
print('The length of serena_williams images is',len(serena_williams))
print('The length of virat_kohli images is',len(virat_kohli))
print("--------------------------------------\n")

dataset=[]
label=[]
img_size=(128,128)

for i , image_name in tqdm(enumerate(lionel_messi),desc="lionel_messi"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/lionel_messi/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(0)


for i , image_name in tqdm(enumerate(maria_sharapova),desc="maria_sharapova"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/maria_sharapova/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(1)



for i , image_name in tqdm(enumerate(roger__federer),desc="roger__federer"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/roger_federer/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(2)

for i , image_name in tqdm(enumerate(serena_williams),desc="serena_williams"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/serena_williams/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(3)
for i , image_name in tqdm(enumerate(virat_kohli),desc="virat_kohli"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/virat_kohli/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_size)
        dataset.append(np.array(image))
        label.append(4)

dataset=np.array(dataset)
label = np.array(label)

print("--------------------------------------\n")
print('Dataset Length: ',len(dataset))
print('Label Length: ',len(label))
print("--------------------------------------\n")

print("--------------------------------------\n")
print("Train-Test Split")
x_train,x_test,y_train,y_test=train_test_split(dataset,label,test_size=0.2,random_state=42)
print("--------------------------------------\n")


print("--------------------------------------\n")
print("Normalaising the Dataset. \n")

# x_train=x_train.astype('float')/255
# x_test=x_test.astype('float')/255 

# Same step above is implemented using tensorflow functions.

x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)

print("--------------------------------------\n")

#model building
cnn_model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dropout(.5),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(5,activation='softmax')
])

cnn_model.summary()
print("--------------------------------------\n")

#model compailing  anf fitting
cnn_model.compile(
  optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])

early_stop =keras.callbacks.EarlyStopping(patience=10,restore_best_weights=10)

history=cnn_model.fit(x_train,y_train,epochs=25,batch_size =32,validation_split=0.1,callbacks=[early_stop])

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

print("Model Evalutaion Phase.\n")
loss,accuracy=cnn_model.evaluate(x_test,y_test)
print(f'Accuracy: {round(accuracy*100,2)}')
print("--------------------------------------\n")
y_pred=cnn_model.predict(x_test)
y_pred = (y_pred > 0.5).astype(int)

print("--------------------------------------\n")
print("Model Prediction.\n")

def make_prediction(img,model):
    img=cv2.imread(img)
    img=Image.fromarray(img)
    img=img.resize((128,128))
    img=np.array(img)
    input_img = np.expand_dims(img, axis=0)
    
    prediction = model.predict(input_img)
    predicted_class = np.argmax(prediction,axis=1)[0]
    class_name = ['lionel_messi','maria_sharapova','roger_federer','serena_williams','virat_kohli']
    predicted_class_name = class_name[predicted_class]
    return predicted_class_name

print(make_prediction('Dataset_Celebrities\cropped\maria_sharapova\maria_sharapova7.png',cnn_model))
print('--------------------------------------------------------')