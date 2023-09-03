# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 21:04:42 2023

@author: doguk
"""
import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import save_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau

#Path
data_dir = '.\HandwrittenNum'

#User-defined parameters
n_epochs = int(input("Enter number of epochs: "))
batch_size = int(input("Enter batch size: "))

# Hyperparameters
z_dim = 100
img_shape = (28, 28, 3)
num_classes = 10  # Number of folders/labels

#Splitting data
datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)

train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(28, 28),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(28, 28),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)
#plot a sample image
sample_training_images, _ = next(train_gen)

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    axes.imshow(images_arr[0])
    axes.axis('off')
    plt.tight_layout()
    plt.show()

plotImages(sample_training_images[:1])

#Create Generator
def build_cgenerator(z_dim, num_classes):
    noise = layers.Input(shape=(z_dim,))
    label = layers.Input(shape=(num_classes,))
    
    # Concatenate noise and label
    merged_input = layers.Concatenate(axis=-1)([noise, label])
    
    g = layers.Dense(128)(merged_input)
    g = layers.LeakyReLU()(g)
    g = layers.Dense(512)(g)
    g = layers.LeakyReLU()(g)
    g = layers.Dense(np.prod(img_shape), activation='tanh')(g)
    img = layers.Reshape(img_shape)(g)
    
    return tf.keras.Model([noise, label], img)

#Create Discriminator
def build_cdiscriminator(img_shape, num_classes):
    img = layers.Input(shape=img_shape)
    label = layers.Input(shape=(num_classes,))
    
    # Flatten the image and concatenate it with the label
    flat_img = layers.Flatten()(img)
    concat = layers.Concatenate(axis=-1)([flat_img, label])
    
    d = layers.Dense(128)(concat)
    d = layers.LeakyReLU()(d)
    d = layers.Dropout(0.25)(d)
    validity = layers.Dense(1, activation='sigmoid')(d)
    
    return tf.keras.Model([img, label], validity)

#Build c-GAN
def build_gan(generator, discriminator):
    noise = layers.Input(shape=(z_dim,))
    label = layers.Input(shape=(num_classes,))
    
    # Concatenate noise and label
    img = generator([noise, label])
    
    discriminator.trainable = False    
    validity = discriminator([img, label])
    
    return tf.keras.Model([noise, label], validity)

# System Monitoring
class SystemMonitor(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print("Beginning of epoch:", epoch)
        self.start_time = time.time()

    def on_epoch_end(self, epoch, count, logs=None):
        elapsed_time = time.time() - self.start_time
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        print(f"\n Epoch: {epoch + 1}/{n_epochs}")
        print(f"Time for Epoch: {elapsed_time:.2f} seconds")
        print(f"CPU Usage: {cpu_percent}%")
        print(f"RAM Usage: {ram_percent}%")
        print(f"Trained Image Count: {count}")

#Build and compile the Discriminator
cdiscriminator = build_cdiscriminator(img_shape, num_classes)
cdiscriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Build and compile the Generator
cgenerator = build_cgenerator(z_dim, num_classes)
cgenerator.compile(loss='binary_crossentropy', optimizer='adam')

#Build and compile the c-GAN
cdiscriminator.trainable = False
cgan = build_gan(cgenerator, cdiscriminator)
cgan.compile(loss='binary_crossentropy', optimizer='adam')


#Additional callback: Reduce learning rate when 'val_loss' has stopped improving
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0004, verbose=1)

# Initialize lists for plotting
g_accs = []
d_accs = []
g_losses = []
d_losses = []
counter = 0
# Initialize best loss for generator
best_g_loss = float('inf')
# Initialize callbacks
system_monitor = SystemMonitor()
system_monitor.set_model(cgan)


# Training loop
for epoch in range(n_epochs):
    system_monitor.on_epoch_begin(epoch)  # Call on_epoch_begin for SystemMonitor
    counter = 0
    #refresh the image counter
    for _ in range(train_gen.__len__()):
        # Get a batch of real images and their corresponding labels
        real_images, labels = train_gen.next()
        
        # Generate a batch of fake images
        noise = np.random.normal(0, 1, (batch_size, z_dim))
        fake_images = cgenerator.predict([noise, labels])
        
        # Labels for real and fake images
        real_y = np.ones((batch_size, 1))
        fake_y = np.zeros((batch_size, 1))
        
        # Train the Discriminator
        d_loss_real = cdiscriminator.train_on_batch([real_images, labels], real_y)
        d_loss_fake = cdiscriminator.train_on_batch([fake_images, labels], fake_y)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train the Generator
        noise = np.random.normal(0, 1, (batch_size, z_dim))
        g_loss = cgan.train_on_batch([noise, labels], real_y)
        
        # Update lists for plotting
        d_losses.append(d_loss[0])
        g_losses.append(g_loss)
        d_accs.append(d_loss[1])
        g_accs.append(d_loss[1])  #Generator accuracy is same as discriminator for fake images
        # Check if the generator loss has improved
        if g_loss < best_g_loss:
            best_g_loss = g_loss
            save_model(cgenerator, 'best_generator_model.h5', overwrite=True)  # Save the best model
        
        # System Monitoring
        counter += 1    
        system_monitor.on_epoch_end(epoch, counter)
        

# Plotting
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(g_accs, label='Generator Accuracy')
plt.plot(d_accs, label='Discriminator Accuracy')
plt.title('Generator and Discriminator Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(g_losses, label='Generator Loss')
plt.plot(d_losses, label='Discriminator Loss')
plt.title('Generator and Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()