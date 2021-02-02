from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np



def build_generator():

    model = Sequential()

    model.add(Dense(256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))
    model.summary()
    
    noise = Input(shape=(latent_dim,))
    img = model(noise)
    return Model(noise, img)

def build_discriminator():
    model = Sequential()

    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    img = Input(shape=img_shape)
    validity = model(img)
    return Model(img, validity)

def train(epochs, batch_size=128, sample_interval=50):
    (X_train, _), (_, _) = mnist.load_data()

    # what is this step about?
    X_train = X_train / 127.5 - 1.

    X_train = np.expand_dims(X_train, axis=3)
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    epoch = 0
    for epoch in range(epochs):
        # generate a set of random indices of the training data
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = combined.train_on_batch(noise, valid)
        gen_imgs = generator.predict(noise)
        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))
        if epoch % sample_interval == 0:
            sample_images(epoch)

def sample_images(epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, latent_dim))
    gen_imgs = generator.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig("images/%d.png" % epoch)
    plt.close()




img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)
latent_dim = 100
optimizer = Adam(0.0002, 0.5)
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy',
                            optimizer=optimizer,
                            metrics=['accuracy'])
generator = build_generator()
z = Input(shape=(latent_dim,))
img = generator(z)
discriminator.trainable = False

validity = discriminator(img)
combined = Model(z, validity)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)

dir(combined)
combined.summary()


epochs=101
batch_size=132
sample_interval=10

(X_train, _), (_, _) = mnist.load_data()

# what is this step about?
X_train = X_train / 127.5 - 1.

X_train = np.expand_dims(X_train, axis=3)
valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

epoch = 0
for epoch in range(epochs):
    # generate a set of random indices of the training data
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    imgs = X_train[idx]
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    # figure this out...
    # train combined model [noise] - [ones]?
    # ones out of the discriminator?
    g_loss = combined.train_on_batch(noise, valid)

    # generate the images
    gen_imgs = generator.predict(noise)
    # train discriminator on real [real imgs] -> [ones]
    d_loss_real = discriminator.train_on_batch(imgs, valid)
    # train discriminator on fake [fake imgs] -> [zeros]
    d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
    # mean loss
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))
    if epoch % sample_interval == 0:
        sample_images(epoch)



if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=101, batch_size=132, sample_interval=10)
    # gan.train(epochs=100000, batch_size=132, sample_interval=10000)



def build_generator_empty():

    model = Sequential()

    model.add(Dense(256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))
    model.summary()
    
    noise = Input(shape=(latent_dim,))
    img = model(noise)
    return Model(noise, img)




generator = build_generator()

# Loads the weights
generator.load_weights("models/epoch_93200/")
r, c = 5, 5
latent_dim = 100
noise = np.random.normal(0, 1, (r * c, latent_dim))

np.random.normal

noise[1]


fig, axs = plt.subplots(r, c)
cnt = 0
for i in range(r):
    for j in range(c):
        axs[i, j].imshow(noise[cnt], cmap='gray')
        axs[i, j].axis('off')
        cnt += 1



gen_imgs = generator.predict(noise)
gen_imgs = 0.5 * gen_imgs + 0.5
fig, axs = plt.subplots(r, c)
cnt = 0
for i in range(r):
    for j in range(c):
        axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
        axs[i, j].axis('off')
        cnt += 1
