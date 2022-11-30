# Rock Paper Scissors Game

Machine Learning Project for Rock Paper Scissors

**BIG THANKS TO:** [SouravJohar](https://github.com/SouravJohar) for the GUI and Gathering Images code!

Prepare the environment


```python
import tensorflow as tf
```

Extract data


```python
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive
    


```python
import os
base_dir = '/content/drive/MyDrive/rockpaperscissors'
```

Checking extracted directory


```python
os.listdir(base_dir)
```




    ['none', 'paper', 'rock', 'scissors']



Data Preprocessing, Auto-labelling, and Image Augmentation


```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=20,
                    horizontal_flip=True,
                    shear_range = 0.2,
                    fill_mode = 'nearest',
                    validation_split=0.4) #set validation split to 40% from total data
```


```python
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(227, 227), 
    batch_size=32,
    color_mode='rgb',
    class_mode='categorical',
    shuffle=True,
    seed=341, #random seed for shuffling and transformations
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    base_dir, # same directory as training data
    target_size=(227, 227),
    batch_size=32,
    color_mode='rgb',
    class_mode='categorical',
    shuffle=True,
    seed=341,
    subset='validation') # set as validation data
```

    Found 480 images belonging to 4 classes.
    Found 320 images belonging to 4 classes.
    

Building Convolutional Neural Network (CNN) model


```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(227, 227, 3)), #Since the image is in RGB, there are 3 channels, 'R', 'G', and 'B'.
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2), #To minimize overfitting
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])
```


```python
#Check model summary
model.summary()
```

    Model: "sequential_2"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d_6 (Conv2D)           (None, 225, 225, 32)      896       
                                                                     
     max_pooling2d_6 (MaxPooling  (None, 112, 112, 32)     0         
     2D)                                                             
                                                                     
     conv2d_7 (Conv2D)           (None, 110, 110, 64)      18496     
                                                                     
     max_pooling2d_7 (MaxPooling  (None, 55, 55, 64)       0         
     2D)                                                             
                                                                     
     conv2d_8 (Conv2D)           (None, 53, 53, 128)       73856     
                                                                     
     max_pooling2d_8 (MaxPooling  (None, 26, 26, 128)      0         
     2D)                                                             
                                                                     
     flatten_2 (Flatten)         (None, 86528)             0         
                                                                     
     dropout_2 (Dropout)         (None, 86528)             0         
                                                                     
     dense_4 (Dense)             (None, 128)               11075712  
                                                                     
     dense_5 (Dense)             (None, 4)                 516       
                                                                     
    =================================================================
    Total params: 11,169,476
    Trainable params: 11,169,476
    Non-trainable params: 0
    _________________________________________________________________
    


```python
#Compile model with 'adam' optimizer and 'categorical_crossentropy' loss function
model.compile(loss='categorical_crossentropy',
              optimizer=tf.optimizers.Adam(),
              metrics=['accuracy'])
```

Create a custom callback that stop the training when accuracy reach 99%


```python
class CallbackThreshold(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(CallbackThreshold, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs={}): 
        accuracy = logs.get('accuracy')
        if accuracy is not None and accuracy >= self.threshold:
          print("\nReached %2.2f%% accuracy, so stopping training." %(self.threshold*100)),
          self.model.stop_training = True

callback = CallbackThreshold(threshold=0.99)
```

Fit and train the model


```python
history = model.fit(
      train_generator,
      steps_per_epoch=15,
      epochs=20,
      validation_data=validation_generator,
      validation_steps=10,
      callbacks=[callback],
      verbose=1)
```

    Epoch 1/20
    15/15 [==============================] - 237s 16s/step - loss: 1.6067 - accuracy: 0.2229 - val_loss: 1.3050 - val_accuracy: 0.4844
    Epoch 2/20
    15/15 [==============================] - 66s 4s/step - loss: 1.0715 - accuracy: 0.5771 - val_loss: 0.7320 - val_accuracy: 0.8281
    Epoch 3/20
    15/15 [==============================] - 65s 4s/step - loss: 0.5680 - accuracy: 0.8062 - val_loss: 0.4326 - val_accuracy: 0.9000
    Epoch 4/20
    15/15 [==============================] - 64s 4s/step - loss: 0.1637 - accuracy: 0.9583 - val_loss: 0.2209 - val_accuracy: 0.9469
    Epoch 5/20
    15/15 [==============================] - 67s 4s/step - loss: 0.0693 - accuracy: 0.9812 - val_loss: 0.1805 - val_accuracy: 0.9344
    Epoch 6/20
    15/15 [==============================] - ETA: 0s - loss: 0.0295 - accuracy: 0.9937
    Reached 99.00% accuracy, so stopping training.
    15/15 [==============================] - 68s 5s/step - loss: 0.0295 - accuracy: 0.9937 - val_loss: 0.3578 - val_accuracy: 0.8687
    

Testing the model


```python
import numpy as np
from google.colab import files
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

uploaded = files.upload()

for fn in uploaded.keys():
 
# predicting images
  path = fn
  img = image.load_img(path, target_size=(227,227))

  imgplot = plt.imshow(img)
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  images = np.vstack([x])


  classes = model.predict(images, batch_size=10)  
  print(fn)
  print(classes)
  if classes[0,0] != 0:
   print('None')
  elif classes[0,1] != 0:
   print('Paper')
  elif classes[0,2] != 0:
    print('Rock')
  else:
    print('Scissors')
```



     <input type="file" id="files-10040b8c-611c-41c6-8ee2-0c755b7740fa" name="files[]" multiple disabled
        style="border:none" />
     <output id="result-10040b8c-611c-41c6-8ee2-0c755b7740fa">
      Upload widget is only available when the cell has been executed in the
      current browser session. Please rerun this cell to enable.
      </output>
      <script>// Copyright 2017 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @fileoverview Helpers for google.colab Python module.
 */
(function(scope) {
function span(text, styleAttributes = {}) {
  const element = document.createElement('span');
  element.textContent = text;
  for (const key of Object.keys(styleAttributes)) {
    element.style[key] = styleAttributes[key];
  }
  return element;
}

// Max number of bytes which will be uploaded at a time.
const MAX_PAYLOAD_SIZE = 100 * 1024;

function _uploadFiles(inputId, outputId) {
  const steps = uploadFilesStep(inputId, outputId);
  const outputElement = document.getElementById(outputId);
  // Cache steps on the outputElement to make it available for the next call
  // to uploadFilesContinue from Python.
  outputElement.steps = steps;

  return _uploadFilesContinue(outputId);
}

// This is roughly an async generator (not supported in the browser yet),
// where there are multiple asynchronous steps and the Python side is going
// to poll for completion of each step.
// This uses a Promise to block the python side on completion of each step,
// then passes the result of the previous step as the input to the next step.
function _uploadFilesContinue(outputId) {
  const outputElement = document.getElementById(outputId);
  const steps = outputElement.steps;

  const next = steps.next(outputElement.lastPromiseValue);
  return Promise.resolve(next.value.promise).then((value) => {
    // Cache the last promise value to make it available to the next
    // step of the generator.
    outputElement.lastPromiseValue = value;
    return next.value.response;
  });
}

/**
 * Generator function which is called between each async step of the upload
 * process.
 * @param {string} inputId Element ID of the input file picker element.
 * @param {string} outputId Element ID of the output display.
 * @return {!Iterable<!Object>} Iterable of next steps.
 */
function* uploadFilesStep(inputId, outputId) {
  const inputElement = document.getElementById(inputId);
  inputElement.disabled = false;

  const outputElement = document.getElementById(outputId);
  outputElement.innerHTML = '';

  const pickedPromise = new Promise((resolve) => {
    inputElement.addEventListener('change', (e) => {
      resolve(e.target.files);
    });
  });

  const cancel = document.createElement('button');
  inputElement.parentElement.appendChild(cancel);
  cancel.textContent = 'Cancel upload';
  const cancelPromise = new Promise((resolve) => {
    cancel.onclick = () => {
      resolve(null);
    };
  });

  // Wait for the user to pick the files.
  const files = yield {
    promise: Promise.race([pickedPromise, cancelPromise]),
    response: {
      action: 'starting',
    }
  };

  cancel.remove();

  // Disable the input element since further picks are not allowed.
  inputElement.disabled = true;

  if (!files) {
    return {
      response: {
        action: 'complete',
      }
    };
  }

  for (const file of files) {
    const li = document.createElement('li');
    li.append(span(file.name, {fontWeight: 'bold'}));
    li.append(span(
        `(${file.type || 'n/a'}) - ${file.size} bytes, ` +
        `last modified: ${
            file.lastModifiedDate ? file.lastModifiedDate.toLocaleDateString() :
                                    'n/a'} - `));
    const percent = span('0% done');
    li.appendChild(percent);

    outputElement.appendChild(li);

    const fileDataPromise = new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        resolve(e.target.result);
      };
      reader.readAsArrayBuffer(file);
    });
    // Wait for the data to be ready.
    let fileData = yield {
      promise: fileDataPromise,
      response: {
        action: 'continue',
      }
    };

    // Use a chunked sending to avoid message size limits. See b/62115660.
    let position = 0;
    do {
      const length = Math.min(fileData.byteLength - position, MAX_PAYLOAD_SIZE);
      const chunk = new Uint8Array(fileData, position, length);
      position += length;

      const base64 = btoa(String.fromCharCode.apply(null, chunk));
      yield {
        response: {
          action: 'append',
          file: file.name,
          data: base64,
        },
      };

      let percentDone = fileData.byteLength === 0 ?
          100 :
          Math.round((position / fileData.byteLength) * 100);
      percent.textContent = `${percentDone}% done`;

    } while (position < fileData.byteLength);
  }

  // All done.
  yield {
    response: {
      action: 'complete',
    }
  };
}

scope.google = scope.google || {};
scope.google.colab = scope.google.colab || {};
scope.google.colab._files = {
  _uploadFiles,
  _uploadFilesContinue,
};
})(self);
</script> 


    Saving 13.jpg to 13.jpg
    1/1 [==============================] - 0s 48ms/step
    13.jpg
    [[0. 0. 1. 0.]]
    Rock
    


    
![png](rock.png)
    


Saving the model


```python
model.save('rock-paper-scissors-model.h5')
```
