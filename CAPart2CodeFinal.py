import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time
import cv2
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Setting global seed
tf.random.set_seed(41)

# Image Augmentation
datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                             zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')


def generate_dataset_PIL(image_directory):
    for f in glob.iglob(image_directory):
        if f.endswith('.jpg'):
            image = load_img(f)
            x = img_to_array(image)
            x = x.reshape((1,) + x.shape)
            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=image_directory[:-1],
                                      save_prefix=f.partition("\\")[2].partition("_")[0],
                                      save_format='jpeg'):
                i += 1
                if (f.partition("\\")[2].partition("_")[0] == 'mixed'):
                    if (i > 24):
                        break
                elif (i > 8):
                    break


# Import image method (including resize)
def import_images(image_directory):
    image_op = []
    class_op = []
    for f in glob.iglob(image_directory):
        if f.endswith('.jpg') | f.endswith('.jpeg'):
            image = cv2.imread(f)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_op.append(cv2.resize(image, dsize=(100, 100), interpolation=cv2.INTER_CUBIC))
            class_op.append(f.partition("\\")[2].partition("_")[0])

    return image_op, class_op


# Data preprocessing
def preprocess(_x_train, _x_test, _y_train, _y_test):
    _x_train = np.asarray(_x_train)
    _x_test = np.asarray(_x_test)
    x_train = np.reshape(_x_train, (_x_train.shape[0], 100, 100, 3))
    x_test = np.reshape(_x_test, (_x_test.shape[0], 100, 100, 3))
    x_train = x_train / 255
    x_test = x_test / 255

    # convert y values into one-hot encoded values
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(_y_train)
    y_train = tf.keras.utils.to_categorical(y_train, 4)
    y_test = label_encoder.fit_transform(_y_test)
    y_test = tf.keras.utils.to_categorical(y_test, 4)

    # check one-hot encoding outcome
    # print(y_train[0:])
    # print(y_train.shape)

    return x_train, x_test, y_train, y_test


# Prepare CNN model
def run_cnn26(_x_train, _x_test, _y_train, _y_test, _x_valid, _y_valid):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(20, (3, 3),
                                     activation='relu', input_shape=(100, 100, 3)))
    model.add(tf.keras.layers.Conv2D(20, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    history = model.fit(_x_train, _y_train,
              batch_size=32, epochs=10, verbose=1,
              validation_data=(_x_valid, _y_valid))

    # Evaluate model
    score = model.evaluate(_x_test, _y_test)
    print("score =", score)

    # Plot model

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(10)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    # Predict Test set results
    labels = ['apple', 'banana', 'mixed', 'orange']
    y_pred = model.predict(_x_test)

    lb = LabelBinarizer()

    lb.fit(labels)
    predict_class = lb.inverse_transform(y_pred)
    test_y = lb.inverse_transform(_y_test)
    print(np.concatenate((predict_class.reshape(len(predict_class), 1), test_y.reshape(len(test_y), 1)), 1))
    print(classification_report(test_y, predict_class))


# Method call to generate images for augmentation
# generate_dataset_PIL("insert file location here")

# Import images and export train_class from train_images
train_x, train_y = import_images("insert file location here")

test_x, test_y = import_images("insert file location here")

# Preprocess outcome
train_x, test_x, train_y, test_y = preprocess(train_x, test_x, train_y, test_y)

# Split data into validation set and training set (done due to number of images generated with image augmentation)
x_train, x_valid, y_train, y_valid = train_test_split(train_x, train_y, random_state=2, test_size=0.2)

# Running of CNN model
start_time = time.time()
run_cnn26(x_train, test_x, y_train, test_y, x_valid, y_valid)
print('Seconds taken: ', time.time() - start_time)