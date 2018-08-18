import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Read data
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

samples.pop(0)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

correction_factor = 0.2
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for batch_sample in batch_samples:
                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                bgr_img = cv2.imread(name)
                b, g, r = cv2.split(bgr_img)
                rgb_img = cv2.merge([r, g, b]) 
                images.append(rgb_img)
                measurement = float(line[3])
                measurements.append(measurement)

                name = './data/IMG/'+batch_sample[1].split('/')[-1]
                bgr_img = cv2.imread(name)
                b, g, r = cv2.split(bgr_img)
                rgb_img = cv2.merge([r, g, b]) 
                images.append(rgb_img)
                measurement = float(line[3]) + correction_factor
                measurements.append(measurement)
          
                name = './data/IMG/'+batch_sample[2].split('/')[-1]
                bgr_img = cv2.imread(name)
                b, g, r = cv2.split(bgr_img)
                rgb_img = cv2.merge([r, g, b]) 
                images.append(rgb_img)
                measurement = float(line[3]) - correction_factor
                measurements.append(measurement)
    
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)
            
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Build model
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

# Train model
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), \
                    validation_data=validation_generator, nb_val_samples=len(validation_samples), \
                    nb_epoch=5)

model.save('model_test.h5')




# import csv
# import cv2
# import numpy as np
# from keras.models import Sequential
# from keras.layers import Flatten, Dense, Lambda, Cropping2D
# from keras.layers.convolutional import Convolution2D
# from keras.layers.pooling import MaxPooling2D

# # Read data
# lines = []
# with open('./data/driving_log.csv') as csvfile:
#     reader = csv.reader(csvfile)
#     for line in reader:
#         lines.append(line)

# images = []
# measurements = []
# lines.pop(0)
# correction_factor = 0.2
# for line in lines:
#     source_path = line[0]
#     filename = source_path.split('/')[-1]
#     current_path = './data/IMG/' + filename
#     bgr_image = cv2.imread(current_path)
#     b, g, r = cv2.split(bgr_image)
#     rgb_image = cv2.merge([r, g, b]) 
#     images.append(rgb_image)
#     measurement = float(line[3])
#     measurements.append(measurement)
    
#     source_path = line[1]
#     filename = source_path.split('/')[-1]
#     current_path = './data/IMG/' + filename
#     bgr_image = cv2.imread(current_path)
#     b, g, r = cv2.split(bgr_image)
#     rgb_image = cv2.merge([r, g, b]) 
#     images.append(rgb_image)
#     measurement = float(line[3]) + correction_factor
#     measurements.append(measurement)
    
#     source_path = line[2]
#     filename = source_path.split('/')[-1]
#     current_path = './data/IMG/' + filename
#     bgr_image = cv2.imread(current_path)
#     b, g, r = cv2.split(bgr_image)
#     rgb_image = cv2.merge([r, g, b]) 
#     images.append(rgb_image)
#     measurement = float(line[3]) - correction_factor
#     measurements.append(measurement)
   
# X_train = np.array(images)
# y_train = np.array(measurements)

# # Build model
# model = Sequential()
# model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
# model.add(Cropping2D(cropping=((70,25),(0,0))))
# model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
# model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
# model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
# model.add(Convolution2D(64,3,3,activation="relu"))
# model.add(Convolution2D(64,3,3,activation="relu"))
# model.add(Flatten())
# model.add(Dense(100))
# model.add(Dense(50))
# model.add(Dense(1))

# # Train model
# model.compile(loss='mse', optimizer='adam')
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5, verbose=1)

# model.save('model.h5')



