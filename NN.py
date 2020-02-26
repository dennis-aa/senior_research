import tensorflow 
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import os
import shutil
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix


'''

The first section sorts the data so it is usable for training
It divides the dataset into training and validation and palces them in the appropriate folder

'''
base_dir = os.path.join('', 'input')

print(base_dir)
model = keras.Sequential()

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'dermatofibroma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

#Identify and locate the image based on ID in csv file
train_dir = os.path.join(base_dir, 'train_dir')

val_dir = os.path.join(base_dir, 'val_dir')

df_data = pd.read_csv(os.path.join(base_dir, 'HAM10000_metadata.csv'))

df = df_data.groupby('lesion_id').count()
df = df[df['image_id'] == 1]

df.reset_index(inplace=True)

print(df.head())

def identify_duplicates(x):
    
    unique_list = list(df['lesion_id'])
    
    if x in unique_list:
        return 'no_duplicates'
    else:
        return 'has_duplicates'
    
df_data['duplicates'] = df_data['lesion_id']
df_data['duplicates'] = df_data['duplicates'].apply(identify_duplicates)

print(df_data.head())

df = df_data[df_data['duplicates'] == 'no_duplicates']

print(df.shape)

#Split data for validation
y = df['dx']

_, df_val = train_test_split(df, test_size=0.17, random_state=101, stratify=y)

df_val.shape

def identify_val_rows(x):
    # create a list of all the lesion_id's in the val set
    val_list = list(df_val['image_id'])
    
    if str(x) in val_list:
        return 'val'
    else:
        return 'train'

df_data['train_or_val'] = df_data['image_id']
# apply the function to this new column
df_data['train_or_val'] = df_data['train_or_val'].apply(identify_val_rows)
   
# filter out train rows
df_train = df_data[df_data['train_or_val'] == 'train']

print(len(df_train))
print(len(df_val))

df_data.set_index('image_id', inplace=True)


# Create file structure to be used for training
folder_1 = os.listdir('input/ham10000_images_part_1')
folder_2 = os.listdir('input/ham10000_images_part_2')

train_list = list(df_train['image_id'])
val_list = list(df_val['image_id'])

for image in train_list:
    
    fname = image + '.jpg'
    label = df_data.loc[image,'dx']
    
    if fname in folder_1:
        # source path to image
        src = os.path.join('input/ham10000_images_part_1', fname)
        # destination path to image
        dst = os.path.join(train_dir, label, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)

    if fname in folder_2:
        # source path to image
        src = os.path.join('input/ham10000_images_part_2', fname)
        # destination path to image
        dst = os.path.join(train_dir, label, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)


# Transfer the val images

for image in val_list:
    
    fname = image + '.jpg'
    label = df_data.loc[image,'dx']
    
    if fname in folder_1:
        # source path to image
        src = os.path.join('input/ham10000_images_part_1', fname)
        # destination path to image
        dst = os.path.join(val_dir, label, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)

    if fname in folder_2:
        # source path to image
        src = os.path.join('input/ham10000_images_part_2', fname)
        # destination path to image
        dst = os.path.join(val_dir, label, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)

#Confirm file spliting
print("\ntrain_dir\n")
print('nv - ', len(os.listdir('input/train_dir/nv')))
print('mel - ',len(os.listdir('input/train_dir/mel')))
print('bkl - ',len(os.listdir('input/train_dir/bkl')))
print('bcc - ',len(os.listdir('input/train_dir/bcc')))
print('akiec - ',len(os.listdir('input/train_dir/akiec')))
print('vasc - ',len(os.listdir('input/train_dir/vasc')))
print('df - ',len(os.listdir('input/train_dir/df')))

print("***************************************")
print('\nval_dir\n')
print('nv - ',len(os.listdir('input/val_dir/nv')))
print('mel - ',len(os.listdir('input/val_dir/mel')))
print('bkl - ',len(os.listdir('input/val_dir/bkl')))
print('bcc - ',len(os.listdir('input/val_dir/bcc')))
print('akiec - ',len(os.listdir('input/val_dir/akiec')))
print('vasc - ',len(os.listdir('input/val_dir/vasc')))
print('df - ',len(os.listdir('input/val_dir/df')))


'''
Following section sets up the model for training

'''

train_path = 'input/train_dir'
valid_path = 'input/val_dir'

num_train_samples = len(df_train)
num_val_samples = len(df_val)
train_batch_size = 10
val_batch_size = 10
image_size = 224

train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)

datagen = ImageDataGenerator(
    preprocessing_function= \
    tensorflow.keras.applications.mobilenet.preprocess_input)

train_batches = datagen.flow_from_directory(train_path,
                                            target_size=(image_size,image_size),
                                            batch_size=train_batch_size)

valid_batches = datagen.flow_from_directory(valid_path,
                                            target_size=(image_size,image_size),
                                            batch_size=val_batch_size)

test_batches = datagen.flow_from_directory(valid_path,
                                            target_size=(image_size,image_size),
                                            batch_size=1,
                                            shuffle=False)


model = keras.Sequential()
#add model layers
model.add(keras.layers.InputLayer(input_shape=(224,224,3)))
model.add(keras.layers.ZeroPadding2D())
model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.ReLU())
model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.ReLU())
model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.ReLU())
model.add(keras.layers.GlobalAveragePooling2D())
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(7, activation='softmax'))
print(model.summary())

from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

model.compile(Adam(lr=0.01), loss='categorical_crossentropy', 
              metrics=[categorical_accuracy, top_2_accuracy, top_3_accuracy])

print(valid_batches.class_indices)

# Try to make the model more sensitive to Nv due to the unbalanced nature of the data.
class_weights={
    0: 1.0, # akiec
    1: 1.0, # bcc
    2: 1.0, # bkl
    3: 1.0, # df
    4: 1.0, # mel 
    5: 3.0, # nv 
    6: 1.0, # vasc
}
filepath = "saved_model.h5"

checkpoint = ModelCheckpoint(filepath, monitor='val_top_3_accuracy', verbose=1, 
                             save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_top_3_accuracy', factor=0.5, patience=2, 
                                   verbose=1, mode='max', min_lr=0.00001)
                              
                              
callbacks_list = [checkpoint, reduce_lr]

history = model.fit_generator(valid_batches, steps_per_epoch=train_steps, 
                              class_weight=class_weights,
                    validation_data=valid_batches,
                    validation_steps=val_steps,
                    epochs=1, verbose=1,
                   callbacks=callbacks_list)