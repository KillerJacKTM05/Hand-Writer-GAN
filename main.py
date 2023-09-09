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


def residual_block(x, filters):
    res = layers.Conv2D(filters, kernel_size=3, padding='same')(x)
    res = layers.BatchNormalization()(res)
    res = layers.LeakyReLU()(res)
    
    res = layers.Conv2D(filters, kernel_size=3, padding='same')(res)
    res = layers.BatchNormalization()(res)
    
    return layers.Add()([res, x])

#Create Generator
def build_cgenerator(z_dim, num_classes, img_shape):
    noise = layers.Input(shape=(z_dim,))
    label = layers.Input(shape=(num_classes,))
    
    merged_input = layers.Concatenate(axis=-1)([noise, label])
    
    g = layers.Dense(7 * 7 * 256)(merged_input)
    g = layers.Reshape((7, 7, 256))(g)
    
    # Two Conv2DTranspose layers for upsampling
    g = layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same')(g)
    g = layers.BatchNormalization()(g)
    g = layers.LeakyReLU()(g)
    
    g = layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same')(g)
    g = layers.BatchNormalization()(g)
    g = layers.LeakyReLU()(g)
    
    # Six residual blocks
    for _ in range(6):
        g = residual_block(g, 64)
    
    # Two Conv2DTranspose layers for upsampling
    g = layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same')(g)
    g = layers.BatchNormalization()(g)
    g = layers.LeakyReLU()(g)
    
    g = layers.Conv2DTranspose(3, kernel_size=3, strides=2, padding='same', activation='sigmoid')(g)
    g = layers.AveragePooling2D(pool_size=(2, 2))(g)
    g = layers.AveragePooling2D(pool_size=(2, 2))(g)
    
    return tf.keras.Model([noise, label], g)

#Create Discriminator
def build_cdiscriminator(img_shape, num_classes):
    img = layers.Input(shape=img_shape)
    label = layers.Input(shape=(num_classes,))
    
    # Six Conv2D LeakyReLU layers
    d = layers.Conv2D(64, kernel_size=3, strides=2, padding='same')(img)
    d = layers.LeakyReLU(alpha=0.01)(d)
    
    d = layers.Conv2D(128, kernel_size=3, strides=2, padding='same')(d)
    d = layers.LeakyReLU(alpha=0.01)(d)
    
    d = layers.Conv2D(256, kernel_size=3, strides=2, padding='same')(d)
    d = layers.LeakyReLU(alpha=0.01)(d)
    
    d = layers.Conv2D(512, kernel_size=3, strides=2, padding='same')(d)
    d = layers.LeakyReLU(alpha=0.01)(d)
    
    d = layers.Conv2D(1024, kernel_size=3, strides=2, padding='same')(d)
    d = layers.LeakyReLU(alpha=0.01)(d)
    
    d = layers.Conv2D(2048, kernel_size=3, strides=2, padding='same')(d)
    d = layers.LeakyReLU(alpha=0.01)(d)
    
    # Flatten and concatenate with label
    d = layers.Flatten()(d)
    concat = layers.Concatenate(axis=-1)([d, label])
    
    # Final dense layer
    validity = layers.Dense(1, activation='sigmoid')(concat)
    
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

    def on_epoch_end(self, epoch, count, g_loss, d_loss, logs=None):
        elapsed_time = time.time() - self.start_time
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        print(f"\n Epoch: {epoch + 1}/{n_epochs}")
        print(f"Time for Epoch: {elapsed_time:.2f} seconds")
        print(f"CPU Usage: {cpu_percent}%")
        print(f"RAM Usage: {ram_percent}%")
        print(f"g loss: {g_loss} d loss: {d_loss}")
        print(f"Trained Image Count: {count}")

# Function to plot generated image
def plot_generated_image(image, label):
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title(f"Generated image with label: {label}")
    plt.axis('off')
    plt.show()
 
# Different Learning Rates (TTUR)
d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0004, beta_1=0.5)
g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5)

#Build and compile the Discriminator
cdiscriminator = build_cdiscriminator(img_shape, num_classes)
cdiscriminator.compile(loss='binary_crossentropy', optimizer=d_optimizer, metrics=['accuracy'])

#Build and compile the Generator
cgenerator = build_cgenerator(z_dim, num_classes, img_shape)
cgenerator.compile(loss='binary_crossentropy', optimizer='adam')

# After defining cgenerator and cdiscriminator
gen_output_shape = cgenerator.output_shape[1:]
disc_input_shape = cdiscriminator.input_shape[0][1:]

if gen_output_shape == disc_input_shape:
    print("Shapes match. You're good to go!")
else:
    print(f"Mismatch! Generator output shape is {gen_output_shape}, but Discriminator input shape is {disc_input_shape}.")

#Build and compile the c-GAN
cdiscriminator.trainable = False
cgan = build_gan(cgenerator, cdiscriminator)
cgan.compile(loss='binary_crossentropy', optimizer=g_optimizer)


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

try:
    # Training loop
    for epoch in range(n_epochs):
        system_monitor.on_epoch_begin(epoch)  # Call on_epoch_begin for SystemMonitor
        counter = 0
        epoch_d_loss = []
        epoch_g_loss = []
        epoch_d_acc = []
        epoch_g_acc = []
        #refresh the image counter
        for _ in range(train_gen.__len__()):
            # Get a batch of real images and their corresponding labels
            real_images, labels = train_gen.next()
        
            # Generate a batch of fake images
            noise = np.random.normal(0, 1, (batch_size, z_dim))
            fake_images = cgenerator.predict([noise, labels])
        
            # Labels for real and fake images
            # Label Smoothing
            real_y = np.ones((batch_size, 1)) * 0.9  # Use 0.9 instead of 1 for real images
            fake_y = np.zeros((batch_size, 1))
        
            # Train the Discriminator
            d_loss_real = cdiscriminator.train_on_batch([real_images, labels], real_y)
            d_loss_fake = cdiscriminator.train_on_batch([fake_images, labels], fake_y)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
            # Train the Generator
            noise = np.random.normal(0, 1, (batch_size, z_dim))
            g_loss = cgan.train_on_batch([noise, labels], real_y)
        
            # Update lists for plotting
            epoch_d_loss.append(d_loss[0])
            epoch_g_loss.append(g_loss)
            epoch_d_acc.append(d_loss[1])
            epoch_g_acc.append(d_loss[1])  #Generator accuracy is same as discriminator for fake images
            # Check if the generator loss has improved
            if g_loss < best_g_loss:
                best_g_loss = g_loss
                save_model(cgenerator, 'best_generator_model.h5', overwrite=True)  # Save the best model
        
            # System Monitoring
            counter += 1    
            system_monitor.on_epoch_end(epoch, counter, g_loss, d_loss)
        
        # Calculate average loss and accuracy for the epoch
        avg_d_loss = np.mean(epoch_d_loss)
        avg_g_loss = np.mean(epoch_g_loss)
        avg_d_acc = np.mean(epoch_d_acc)
        avg_g_acc = np.mean(epoch_g_acc)
        # Append to lists for plotting
        d_losses.append(avg_d_loss)
        g_losses.append(avg_g_loss)
        d_accs.append(avg_d_acc)
        g_accs.append(avg_g_acc)
        # Plot a sample image from the generator
        sample_noise = np.random.normal(0, 1, (1, z_dim))
        random_label_index = np.random.randint(0, num_classes)
        sample_label = np.zeros((1, num_classes))
        sample_label[0, random_label_index] = 1  # One-hot encoding
        generated_image = cgenerator.predict([sample_noise, sample_label])
        plot_generated_image(generated_image[0, :, :, 0], str(random_label_index))
        
except KeyboardInterrupt:
    print("Training interrupted. Saving models.")
    cgenerator.save('generator_model_interrupted.h5')
    cdiscriminator.save('discriminator_model_interrupted.h5')
    cgan.save('gan_model_interrupted.h5')
    
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