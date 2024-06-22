

import numpy as np
import tensorflow as tf
import os
import cv2
import glob
import sklearn
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import Xception
from tensorflow import keras
from keras.layers import Conv2D,MaxPooling2D,BatchNormalization,Activation
from keras.layers import Reshape,Lambda
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow import multiply
from keras import backend as K









#%%


SIZE = 112   
train_images = []
train_labels = [] 
for directory_path in glob.glob("/*"):
    
    label = directory_path.split("\\")[-1]
    print(label) 
    for img_path in glob.glob(os.path.join(directory_path, "/*.jpg")):
        print(img_path)
        img = cv2.imread(img_path,  cv2.IMREAD_COLOR)
        img = cv2.resize(img , (SIZE, SIZE))
        train_images.append(img)
        train_labels.append(label)

train_images = np.array(train_images)
train_labels = np.array(train_labels)


test_images = []
test_labels = [] 
for directory_path in glob.glob("/*"):
    label = directory_path.split("\\")[-1]
    print(label) 
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        print(img_path)
        img = cv2.imread(img_path,  cv2.IMREAD_COLOR)
        img = cv2.resize(img , (SIZE, SIZE))
        test_images.append(img)
        test_labels.append(label)

test_images = np.array(test_images)
test_labels = np.array(test_labels)
#x_train, x_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

x_train, x_test, y_train, y_test = train_images,test_images,train_labels,test_labels

# Normalize pixel values to between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

#label encode
label_encoder = LabelEncoder()

# Fit and transform the y_train labels
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.fit_transform(y_test)
#use label encoding
y_train_one_hot = to_categorical(y_train_encoded)
y_test_one_hot = to_categorical(y_test_encoded)



#%%
def final(Layers):
        Batch_size, H, W, C = Layers.shape
        # Compute channel-wise attention
        channel_attention = tf.reduce_mean(Layers, axis=-1, keepdims=True)
        print(channel_attention.shape)
        # Compute spatial attention
        spatial_attention = tf.keras.layers.Conv2D(filters=1, kernel_size=3, padding='same', activation='sigmoid')(channel_attention)

        # Apply spatial attention to input features
        attended_inputs = tf.multiply(Layers, spatial_attention)
       
        
        conv_3x3 = tf.keras.layers.Conv2D(filters=C, kernel_size=(3, 3), padding='same')(attended_inputs)
        bn_3x3 = tf.keras.layers.BatchNormalization()(conv_3x3)

        # 1x1 Convolution without Batch Normalization
        conv_1x1 = tf.keras.layers.Conv2D(filters=C, kernel_size=(1, 1), padding='same')(attended_inputs)

        # Combine the outputs
        combined_output = tf.keras.layers.Add()([bn_3x3, conv_1x1])
        print(combined_output.shape)
        # Global Average Pooling
        gap = tf.reduce_mean(combined_output, axis=[1, 2], keepdims=True)
    
        # Global Max Pooling
        gmp = tf.reduce_max(combined_output, axis=[1, 2], keepdims=True)
    
        # Dense layer for H dimension
        dense_neurons_h = combined_output.shape[1]
        dense_output_h = tf.keras.layers.Dense(dense_neurons_h)(tf.keras.layers.Flatten()(gap))
        print(dense_output_h.shape)
        # Dense layer for W dimension
        dense_neurons_w = combined_output.shape[2]
        dense_output_w = tf.keras.layers.Dense(dense_neurons_w)(tf.keras.layers.Flatten()(gmp))
        print(dense_output_w.shape)
        multiplied_output = tf.matmul(dense_output_h, dense_output_w, transpose_a=True)
        print(multiplied_output.shape)
        # Expand the dimensions to match the desired shape (H x W x 1)
        multiplied_output = tf.expand_dims(multiplied_output, axis=-1)
        #pass it thorugh sigmoid, multiply with input
        sigmoid_output = Activation('sigmoid')(multiplied_output)
        print(sigmoid_output.shape)
        multiplied_output = Layers* sigmoid_output
        print(multiplied_output.shape)

        return multiplied_output
    


#%%

def custom_feature_map(inputs, k):
    # Get the shape of the input tensor
    Batch_size=32
    Batch_size, H, W, C = inputs.shape

    # Apply a Conv2D layer
    x1 = Conv2D(k * C, 1, padding='same')(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    # Global Average Pooling
    gap = tf.reduce_mean(x1, axis=[1, 2], keepdims=True)  # (Batch_size, 1, 1, k*C)
    print(gap.shape)
    gap = Reshape((1, 1, C, k))(gap)  # (Batch_size, 1, 1, C, k)
    gap = Lambda(lambda x: K.mean(x, axis=-1, keepdims=False))(gap)  # (Batch_size, 1, 1, C)
    
    # Global Max Pooling
    gmp = tf.reduce_max(x1, axis=[1, 2], keepdims=True)  # (Batch_size, 1, 1, k*C)
    gmp = Reshape((1, 1, C, k))(gmp)  # (Batch_size, 1, 1, C, k)
    gmp = Lambda(lambda x: K.mean(x, axis=-1, keepdims=False))(gmp)  # (Batch_size, 1, 1, C)

    # Dense layer for H dimension
    dense_output_h = Dense(H)(tf.keras.layers.Flatten()(gap))  # (Batch_size, H)
    dense_output_h = Reshape((H, 1))(dense_output_h)  # (Batch_size, H, 1)

    # Dense layer for W dimension
    dense_output_w = Dense(W)(tf.keras.layers.Flatten()(gmp))  # (Batch_size, W)
    dense_output_w = Reshape((W, 1))(dense_output_w)  # (Batch_size, W, 1)

    # Compute outer product of dense_output_h and dense_output_w
    multiplied_output = tf.matmul(dense_output_h, dense_output_w, transpose_b=True)  # (Batch_size, H, W)

    # Expand the dimensions to match the desired shape (H x W x 1)
    multiplied_output = tf.expand_dims(multiplied_output, axis=-1)  # (Batch_size, H, W, 1)
    print(multiplied_output.shape)

    # Pass it through sigmoid
    sigmoid_output = tf.sigmoid(multiplied_output)
    print(sigmoid_output.shape)

    # Multiply with the input
    output = inputs * sigmoid_output
    print(output.shape)
    return output

#%%


#%%



#%%


#%%

from tensorflow.keras.layers import Input, UpSampling2D, Concatenate, GlobalAveragePooling2D, Dense


# Define the input shape
input_shape = (112, 112, 3)

input_layer = Input(shape=input_shape)
# Load the EfficientNetV2S model with pre-trained weights, excluding the top layers
x_model = Xception(weights='imagenet', include_top=False, input_tensor=input_layer)


# Create the model
feature1 = x_model.get_layer('block1_conv1 ').output
print(feature1.shape)
feature2 = x_model.get_layer('conv2d_4').output
print(feature2.shape)
feature3 = x_model.get_layer('block4_sepconv2_bn ').output
print(feature3.shape)
feature4 = x_model.get_layer('block13_sepconv2_bn ').output
print(feature4.shape)

# Upsample feature4 to 14x14
feature4_upsampled = UpSampling2D(size=(2, 2))(feature4)  # (None, 14, 14, 728)

# Resize feature maps using UpSampling2D
C1=feature1.shape[3]
feature1_upsampled = Conv2D(filters=C1,kernel_size=(2,2),strides=(2,2),padding='valid')(feature1)
feature1_upsampled =MaxPooling2D(pool_size=(2,2))(feature1_upsampled)
feature2_upsampled =MaxPooling2D(pool_size=(2,2))(feature2)
# Ensure the channels match

#the channel dimensions as well:
feature1_upsampled = Conv2D(1024, (1, 1), padding='same')(feature1_upsampled)
feature2_upsampled = Conv2D(1024, (1, 1), padding='same')(feature2_upsampled)
feature4_upsampled = Conv2D(1024, (1, 1), padding='same')(feature4_upsampled)
print(feature1_upsampled.shape)
print(feature2_upsampled.shape)
print(feature3.shape)
print(feature4_upsampled.shape)

k=5
csa1=custom_feature_map(feature1_upsampled,k)
csa2=custom_feature_map(feature2_upsampled,k)
csa3=custom_feature_map(feature3,k)
csa4=custom_feature_map(feature4_upsampled,k)
f1=final(csa1)
f2=final(csa2)
f3=final(csa3)
f4=final(csa4)
concatenated_features = Concatenate(axis=-1)([f1, f2, f3, f4])

# Add a global average pooling layer
global_avg_pool = GlobalAveragePooling2D()(concatenated_features)

# Add a fully connected layer and output layer
#dense_layer = Dense(256, activation='relu')(global_avg_pool)
output_layer = Dense(5, activation='softmax')(global_avg_pool)

model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
     # set input mean to 0 over the dataset
     featurewise_center=False,
     # set each sample mean to 0
     samplewise_center=False,
     # divide inputs by std of dataseta
     featurewise_std_normalization=False,
     # divide each input by its std
     samplewise_std_normalization=False,
     # apply ZCA whitening
     zca_whitening=False,
     # epsilon for ZCA whitening
     zca_epsilon=1e-06,
     # randomly rotate images in the range (deg 0 to 180)
     rotation_range=20,
     # randomly shift images horizontally
     width_shift_range=0,
     # randomly shift images vertically
     height_shift_range=0,
     # set range for random shear
     shear_range=0.2,       
     # set range for random zoom
     zoom_range=0,
     # set range for random channel shifts
     channel_shift_range=0,
     # set mode for filling points outside the input boundaries
     fill_mode='nearest',
     # value used for fill_mode = "constant"
     cval=0,   
     # randomly flip images
     horizontal_flip=True,
     # randomly flip images
     vertical_flip=True,
     # set rescaling factor (applied before any other transformation)
     rescale=None,
     # set function that will be applied on each input
     preprocessing_function=None,
     # image data format, either "channels_first" or "channels_last"
     data_format=None,
     # fraction of images reserved for validation (strictly between 0 and 1)
     validation_split=0.0)  

 # Compute quantities required for featurewise normalization    
 # (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(x_train)
history= model.fit(datagen.flow(x_train, y_train_one_hot, batch_size =16 ),
                            epochs  = 2 ,validation_data = (x_test, y_test_one_hot))



#%%



#%%


