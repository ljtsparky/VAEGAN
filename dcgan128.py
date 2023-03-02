import os
import time as t
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.inception_score import get_inception_score
from utils.tensorboard_logger import Logger
from itertools import chain
from torchvision import utils
latent_dim = 128
celebA_size = 128
# learning_rate = 0.0001

class Generator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [1024, 512, 256]
        # Output_dim = C (number of channels)
        self.main_module = nn.Sequential(
            # State (latex1x1)
            nn.ConvTranspose2d(in_channels=latent_dim, out_channels=1024, kernel_size=4, stride=1, padding=0,
                               bias=False),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),

            # State (1024 4 4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding=1,
                               bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # State (512 8 8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1,
                               bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),
            # 256 16 16
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1,
                               bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(True),
            # 128 32 32
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1,
                               bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=64, out_channels=channels, kernel_size=5, stride=2, padding=2, output_padding=1,
                               bias=False),
            # 3 64 64
        )
        self.Tanh = nn.Tanh()
    def forward(self, x):  # input (b,latentdim)
        # 64 128 1 1
        x = self.main_module(x)
        # print("decoder finally x shape",x.shape)
        return self.Tanh(x)
class Discriminator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.main_module = nn.Sequential(
            # Image (3x64x64)
            nn.Conv2d(in_channels=channels, out_channels=128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.2,inplace=True),  # do this directly in the memory instead of using another temp memory
            # State (128 32 32)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.LeakyReLU(0.2,inplace=True),
            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(512, momentum=0.9),
            nn.LeakyReLU(0.2,inplace=True),
            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(1024, momentum=0.9),
            nn.LeakyReLU(0.2,inplace=True),
            # State (1024 4 4)
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(2048, momentum=0.9),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=2048, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print("Discriminator before forward size",x.shape)
        x = self.main_module(x)
        # print("Discriminator after forward size", x.shape)
        return x

class DCGAN_MODEL128(object):
    def __init__(self, args):
        print("dcgan ---> vaegan model initalization.")
        self.G = Generator(args.channels)
        self.D = Discriminator(args.channels)
        self.C = args.channels
        self.train_model = args.model
        self.dataset = args.dataset
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        print(  str(self.batch_size)  )
        self.path_to_save = self.train_model + ' ' + self.dataset + ' ' + str(self.epochs) + ' ' + str(self.batch_size)
        # binary cross entropy loss and optimizer
        self.loss = nn.BCELoss()

        self.cuda = False
        self.cuda_index = 0
        # check if cuda is available
        self.check_cuda(args.cuda)
        # Using lower learning rate than suggested by (ADAM authors) lr=0.0002  and Beta_1 = 0.5 instead od 0.9 works better [Radford2015]
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.number_of_images = 10

    # cuda support
    def check_cuda(self, cuda_flag=False):
        if cuda_flag: # default false
            self.cuda = True
            self.D.cuda(self.cuda_index)
            self.G.cuda(self.cuda_index)
            self.loss = nn.BCELoss().cuda(self.cuda_index)
            print("Cuda enabled flag: ",self.cuda)


    def train(self, train_loader):
        self.t_begin = t.time()
        if not os.path.exists('training_result_images/'):
            os.makedirs('training_result_images/')
        self.load_model()
        generator_iter = 218400
        #self.file = open("inception_score_graph.txt", "w")

        for epoch in range(self.epochs):
            self.epoch_start_time = t.time()
            for i, (images, _) in enumerate(train_loader):
                print(i)
                # Check if round number of batches, to prevent not enough data to form a batchsize dataset
                if i == train_loader.dataset.__len__() // self.batch_size:
                    break

                z = torch.randn(self.batch_size, latent_dim,1,1)  # [0,1)
                real_labels = torch.ones(self.batch_size)
                fake_labels = torch.zeros(self.batch_size)
                images, z = Variable(images).cuda(self.cuda_index), Variable(z).cuda(self.cuda_index)
                real_labels, fake_labels = Variable(real_labels).cuda(self.cuda_index), Variable(fake_labels).cuda(
                    self.cuda_index)
                # Train discriminator
                # Compute BCE_Loss using real images
                outputs = self.D(images)
                d_loss_real = self.loss(outputs.flatten(), real_labels)
                real_score = outputs

                # Compute BCE Loss using fake images
                fake_images = self.G(z)
                outputs = self.D(fake_images)
                d_loss_fake = self.loss(outputs.flatten(), fake_labels)
                fake_score = outputs

                # Optimize discriminator
                d_loss = d_loss_real + d_loss_fake
                self.D.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Train generator
                # Compute loss with fake images
                if generator_iter % 5 ==0 :
                    z = Variable(torch.randn(self.batch_size, latent_dim,1,1)).cuda(self.cuda_index)
                    fake_images = self.G(z)
                    outputs = self.D(fake_images)
                    g_loss = self.loss(outputs.flatten(), real_labels)
                    # Optimize generator
                    self.D.zero_grad()
                    self.G.zero_grad()
                    g_loss.backward()
                    self.g_optimizer.step()
                generator_iter += 1

                if generator_iter % 300 == 0:
                    ## inception score###########################
                    sample_list = []
                    for i in range(10):
                        z = Variable(torch.randn(10, latent_dim,1,1)).cuda(self.cuda_index)
                        samples = self.G(z)
                        sample_list.append(samples.data.cpu().numpy())

                    # Flattening list of lists into one list of numpy arrays
                    new_sample_list = list(chain.from_iterable(sample_list))
                    print("Calculating Inception Score over 1k generated images")
                    # Feeding list of numpy arrays
                    inception_score = get_inception_score(new_sample_list, cuda=True, batch_size=32,
                                                          resize=True, splits=10)
                    ## #################################################

                    print('Epoch-{}'.format(epoch + 1))
                    self.save_model()
                    # Denormalize images and save them in grid 8x8
                    z = Variable(torch.randn(64, latent_dim,1,1)).cuda(self.cuda_index)
                    samples = self.G(z)
                    samples = samples.mul(0.5).add(0.5)
                    samples = samples.data.cpu()[:64]
                    grid = utils.make_grid(samples)
                    utils.save_image(grid,
                                     'DCGAN128/randn_z_iter_{}.png'.format(str(generator_iter).zfill(3)))
                    # Denormalize images and save them in grid 8x8

                    time = t.time() - self.t_begin
                    print("Generator iter: {}".format(generator_iter))
                    print("Time {}".format(time))
                    dict = {'iters': [generator_iter], 'genloss': [g_loss.cpu().data.numpy()],
                            'disloss': [d_loss.cpu().data.numpy()], 'inscore': [inception_score[0]]}
                    df = pd.DataFrame(dict)
                    df.to_csv('gallery/dcgan128_loss_inscore_celebA128.csv', mode='a', index=False, header=False)


                if ((i + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (i + 1), train_loader.dataset.__len__() // self.batch_size, d_loss.data, g_loss.data))

        self.t_end = t.time()
        print('Time of training-{}'.format((self.t_end - self.t_begin)))
        #self.file.close()

        # Save the trained parameters
        self.save_model()

    def evaluate(self, test_loader, D_model_path, G_model_path):
        self.load_model(D_model_path, G_model_path)
        z = Variable(torch.randn(self.batch_size, latent_dim, 1, 1)).cuda(self.cuda_index)
        samples = self.G(z)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()
        grid = utils.make_grid(samples)
        print("Grid of 8x8 images saved to 'dgan_model_image.png'.")
        utils.save_image(grid, 'dgan_model_image.png')

    def real_images(self, images, number_of_images):
        if (self.C == 3):
            return self.to_np(images.view(-1, self.C, celebA_size , celebA_size)[:self.number_of_images])
        else:
            return self.to_np(images.view(-1, celebA_size, celebA_size)[:self.number_of_images])

    def generate_img(self, z, number_of_images):
        samples = self.G(z).data.cpu().numpy()[:number_of_images]
        generated_images = []
        for sample in samples:
            if self.C == 3:
                generated_images.append(sample.reshape(self.C, celebA_size, celebA_size))
            else:
                generated_images.append(sample.reshape(celebA_size, celebA_size))
        return generated_images

    def to_np(self, x):
        return x.data.cpu().numpy()

    def save_model(self):
        if not os.path.exists(self.path_to_save):
            os.makedirs(self.path_to_save)
        os.chdir(self.path_to_save)
        torch.save(self.G.state_dict(), './generator.pkl')
        torch.save(self.D.state_dict(), './discriminator.pkl')
        os.chdir('..')
        print('Models save to ./generator.pkl & ./discriminator.pkl ')

    def load_model(self):
        Generator_model_path = os.path.join('DCGAN128 celebA128 30 64', 'generator.pkl') # current working dir + filename
        Discriminator_model_path = os.path.join('DCGAN128 celebA128 30 64', 'discriminator.pkl')
        self.G.load_state_dict(torch.load(Generator_model_path))
        self.D.load_state_dict(torch.load(Discriminator_model_path))

    def generate_latent_walk(self):
        self.load_model()
        print('networks of dcgan generator,loaded')
        # Interpolate between twe noise(z1, z2) with number_int steps between
        number_int = 10
        z_intp = torch.FloatTensor(number_int, latent_dim, 1, 1)
        z1 = torch.randn(number_int, latent_dim, 1, 1)
        z2 = torch.randn(number_int, latent_dim, 1, 1)
        if self.cuda:
            z_intp = z_intp.cuda()
            z1 = z1.cuda()
            z2 = z2.cuda()

        z_intp = Variable(z_intp)
        images = []
        alpha = 1.0 / float(number_int + 1)
        print(alpha)
        for i in range(1, number_int + 1):
            z_intp.data = z1*alpha + z2*(1.0 - alpha)
            alpha += alpha
            fake_im = self.G(z_intp)
            fake_im = fake_im.mul(0.5).add(0.5) #denormalize
            for j in range(number_int):
                temp = fake_im[j]
                images.append(temp.view(self.C, celebA_size, celebA_size).data.cpu())

        grid = utils.make_grid(images, nrow=number_int)
        utils.save_image(grid, 'gallery/latent walk 128/dcgan128 randnz walk.png')
        print("Saved interpolated images to gallery/dcgan rand z walk.png")

    def get_inception_score(self):
        self.load_model()
        sample_list = []
        for i in range(12):
            z = Variable(torch.randn(60, latent_dim,1,1)).cuda(self.cuda_index)
            samples = self.G(z)
            sample_list.append(samples.data.cpu().numpy())

        # Flattening list of lists into one list of numpy arrays
        new_sample_list = list(chain.from_iterable(sample_list))
        print("Calculating Inception Score over 1k generated images")
        # Feeding list of numpy arrays
        inception_score = get_inception_score(new_sample_list, cuda=True, batch_size=32,
                                              resize=True, splits=10)
        dict = {'inscore': [inception_score[0],inception_score[1]]}
        df = pd.DataFrame(dict)
        # df.to_csv('gallery/vaegan128_loss_inscore_horse2zebra.csv', mode='a', index=False, header=False)
        df.to_csv('gallery/dcgan128.csv', mode='a', index=False, header=False)

    def compare_encoders(self, z1, z2):
        self.load_model()
        image2 = self.G(z2.view(-1, latent_dim, 1, 1))
        image1 = self.G(z1.view(-1, latent_dim, 1, 1))
        return image1,image2
