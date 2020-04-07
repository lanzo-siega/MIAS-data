from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

train_image = []
realt = X_train1['ID']

# creating a training dataframe from the image dataset by converting each image 
# an array and appending to train_image

for i in tqdm(realt):
  img = image.load_img('/MIAS/Dataset/Train/{}.pgm'.format(i), 
                       target_size=(28,28,1),grayscale=True)
  img = image.img_to_array(img)
  img = img/255
  train_image.append(img)

# creating new training set from train_image and y_train arrays
X = np.array(train_image)
y = to_categorical(y_train1)

# creating a validation set for the CNN to pass through
val_image = []
valt = X_test1['ID']

for i in tqdm(valt):
  img = image.load_img('/MIAS/Dataset/Validation/{}.pgm'.format(i), 
                       target_size=(28,28,1),grayscale=True)
  img = image.img_to_array(img)
  img = img/255
  val_image.append(img)

valX = np.array(val_image)
valy = to_categorical(y_test1)

# building the model
model = Sequential()

# three convolutional layers
model.add(Conv2D(32, (3, 3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# two flatten and dense layers
model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dense(4, activation='softmax'))

# compiling the data and using the 'accuracy' metric to evaluate the model
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

# fitting the model
hist = model.fit(X, y, batch_size=127, epochs=150, validation_data=(valX, valy))

# graphing the model's accuracy per epoch
print(hist.history.keys())
plt.figure(1)
plt.plot(hist.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.savefig('acc_graph.jpg')
