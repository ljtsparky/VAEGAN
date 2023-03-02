import os
import time as t
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
# from utils.inception_score import get_inception_score
# from utils.tensorboard_logger import Logger
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ExponentialLR,MultiStepLR
from itertools import chain
from torchvision import utils
latent_dim = 128
celebA_size = 128
learning_rate = 3e-4
alpha=0.1
# Thanks to https://github.com/Zeleni9/pytorch-wgan giving me basic idea of my code structures.
class Decoder(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [1024, 512, 256]
        # Output_dim = C (number of channels)
        self.main_module = nn.Sequential(
            # State (256x8x8)
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=5, stride=2, padding=2,output_padding=1,bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # State (256 16 16)
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2,output_padding=1,bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(True),

            # State (128 32 32)
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2,output_padding=1,bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(True),
            # (64 64 64 )
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1,
                               bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(True),
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=latent_dim,out_features=256*8*8,bias=False),
            nn.BatchNorm1d(num_features=256*8*8,momentum=0.9),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Conv2d(in_channels=32 ,out_channels=3 ,kernel_size=5,stride=1,padding=2)
        self.Tanh = nn.Tanh()

    def forward(self, x): #input (b,latentdim)
        x=self.fc(x)
        x=x.view(x.shape[0],-1,8,8)
        x = self.main_module(x)
        x = self.conv(x)
        # print("decoder finally x shape",x.shape)
        return self.Tanh(x)

class Discriminator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.main_module = nn.Sequential(
            # Image (3x128x128)
            nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.ReLU(inplace=True),  # do this directly in the memory instead of using another temp memory

            # State (32x64x64)
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.ReLU(inplace=True),

            # State (128x32x32)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.ReLU(inplace=True),
            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.ReLU(inplace=True),
            # State (256x8x8) STOP HERE
        )
        self.dense1 = nn.Sequential(
            nn.Linear(in_features=256*8*8,out_features=512),
            nn.BatchNorm1d(512,momentum=0.9),
            nn.ReLU(inplace=True)
        )
        self.dense2 = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print("Discriminator before forward size",x.shape)
        x = self.main_module(x)
        x =x.view(x.shape[0],256*8*8)
        # print("Discriminator between forward size", x.shape)
        x=self.dense1(x)
        x1=x
        x=self.dense2(x)
        return x,x1

class Encoder(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.main_module = nn.Sequential(
            # Image (3 128 128)
            nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=5, stride=2, padding=2,bias=False),
            nn.BatchNorm2d(64,momentum=0.9),
            nn.ReLU(inplace=True),# do this directly in the memory instead of using another temp memory

            # State (64x64x64)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2,bias=False),
            nn.BatchNorm2d(128,momentum=0.9),
            nn.ReLU(inplace=True),

            # State (128x32x32)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2,bias=False),
            nn.BatchNorm2d(256,momentum=0.9),
            nn.ReLU(inplace=True),
            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.ReLU(inplace=True),
            # (256 8 8)
        )

        self.dense_256_8_8_1024 = nn.Sequential(nn.Linear(256 * 8 * 8, 2048, bias=False),
                                                 nn.BatchNorm1d(num_features=2048, momentum=0.9),
                                                 nn.ReLU(inplace=True)
                                                 )
        self.output1 =nn.Linear(in_features=2048, out_features=latent_dim)
        self.output2 = nn.Linear(in_features=2048, out_features=latent_dim)


    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_() #e**(x*0.5)
        eps = torch.FloatTensor(std.size()).normal_()
        if torch.cuda.is_available():
            eps = Variable(eps.cuda())
        else:
            eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        # print("Discriminator before forward size",x.shape)
        # (b,3,128,128)
        x = self.main_module(x)
        # (b,256,8,8)
        x = x.view(x.shape[0],-1)
        x= self.dense_256_8_8_1024(x)
        # (b,256,8,8)
        z_mean = self.output1(x)    # z_mean = z_mean.view(z_mean.shape[0],-1,1,1)
        z_log_var = self.output2(x)   # z_log_var = z_log_var.view(z_log_var.shape[0], -1, 1, 1)
        z = self.reparametrize(z_mean,z_log_var)
        return z, z_mean, z_log_var


    # def feature_extraction(self, x):
    #     # Use discriminator for feature extraction then flatten to vector of 16384 features
    #     x = self.main_module(x)
    #     return x.view(-1, 1024*4*4)


reconstruction_function = nn.MSELoss(size_average=False)

class VAEGAN_MODEL128final(object):
    def __init__(self, args):
        print("vaegan model initalization.")
        self.D = Decoder(args.channels)
        self.E = Encoder(args.channels)
        self.Dis = Discriminator(args.channels)
        self.C = args.channels
        self.train_model = args.model
        self.BCEloss = nn.BCELoss()
        self.dataset = args.dataset; self.epochs = args.epochs; self.batch_size = args.batch_size
        self.gamma = 15
        print(  str(self.batch_size)  )
        self.path_to_save = self.train_model + ' ' + self.dataset + ' ' + str(self.epochs) + ' ' + str(self.batch_size)
        #
        self.cuda = False;  self.cuda_index = 0
        # check if cuda is available
        self.check_cuda(args.cuda)

        # Using lower learning rate than suggested by (ADAM authors) lr=0.0002  and Beta_1 = 0.5 instead od 0.9 works better [Radford2015]
        # self.e_optimizer = torch.optim.Adam(self.E.parameters(), lr=0.0005, betas=(0.5, 0.999))
        # self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=0.0005, betas=(0.5, 0.999))
        # self.dis_optimizer = torch.optim.Adam(self.Dis.parameters(),lr=0.0005, betas=(0.5,0.999))
        # self.e_optimizer = torch.optim.RMSprop(self.E.parameters(),lr=learning_rate)
        # self.d_optimizer = torch.optim.RMSprop(self.D.parameters(), lr=learning_rate)
        # self.dis_optimizer = torch.optim.RMSprop(self.Dis.parameters(), lr=learning_rate*alpha)
        self.e_optimizer = RMSprop(params=self.E.parameters(), lr=learning_rate, alpha=0.9, eps=1e-8, weight_decay=0,
                                    momentum=0, centered=False)

        self.lr_encoder = ExponentialLR(self.e_optimizer, gamma=0.75)

        self.d_optimizer = RMSprop(params=self.D.parameters(), lr=learning_rate, alpha=0.9, eps=1e-8, weight_decay=0,
                                    momentum=0, centered=False)
        self.lr_decoder = ExponentialLR(self.d_optimizer, gamma=0.75)

        self.dis_optimizer = RMSprop(params=self.Dis.parameters(), lr=learning_rate, alpha=0.9, eps=1e-8,
                                          weight_decay=0, momentum=0, centered=False)
        self.lr_discriminator = ExponentialLR(self.dis_optimizer, gamma=0.75)
        self.number_of_images = 10


    def loss_function(self, mu, logvar ):
        # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_div_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        # -((mu**2)+(e**logvar))+logvar
        KLD = torch.sum(kl_div_element).mul_(-0.5)
        return KLD

    def check_cuda(self, cuda_flag=False):# cuda support
        if cuda_flag: # default false
            self.cuda = True
            self.E.cuda(self.cuda_index) # encoder
            self.D.cuda(self.cuda_index) # decoder
            self.Dis.cuda(self.cuda_index) #discriminator
            self.BCEloss = nn.BCELoss().cuda(self.cuda_index)
            print("Cuda enabled flag: ",self.cuda)

    def reconstruct_a_image(self, images):
        self.load_model()
        # self.Dis.load_state_dict(torch.load(discriminator_path))
        print('networks of E,D,DIS loaded')
        images= images.cuda(self.cuda_index)
        latent_code_z,_,_=self.E(images)
        sample=self.D(latent_code_z)
        sample=sample.mul(0.5).add(0.5)
        sample = sample.data.cpu()
        grid = utils.make_grid(sample)
        print("Grid of 8x8 images saved to 'vaegan_model_image.png'.")
        utils.save_image(grid, 'gallery/vaegan128_model_image.png')
        sample = images
        sample = sample.mul(0.5).add(0.5)
        sample = sample.data.cpu()
        grid = utils.make_grid(sample)
        print("Grid of 8x8 images saved to 'vaegan128_model_image.png'.")
        utils.save_image(grid, 'gallery/original test loader images.png')

    def single_img(self):
        self.load_model()
        images= images.cuda(self.cuda_index)
        z = torch.randn(self.batch_size, latent_dim)
        sample=self.D(z)
        sample=sample.mul(0.5).add(0.5)
        sample = sample.data.cpu()
        # print(sample.shape)
        grid = utils.make_grid(sample)
        utils.save_image(grid, 'gallery/128x128/VAEGAN128 CelebA128 randnz.png')
        sample = images
        sample = sample.mul(0.5).add(0.5)
        sample = sample.data.cpu()
        grid = utils.make_grid(sample)
        print("Grid of 8x8 images saved to 'vaegan128_model_image.png'.")
        utils.save_image(grid, 'gallery/original test loader images.png')
    def train(self, train_loader):
        self.t_begin = t.time()
        if not os.path.exists('vaegan128final/'):
            os.makedirs('vaegan128final/')
        generator_iter = 2100

        self.load_model()
        for epoch in range(self.epochs):
            self.epoch_start_time = t.time()
            # for i, images in enumerate(train_loader):
            for i, (images,_) in enumerate(train_loader):
                print(i)

                # Check if round number of batches, to prevent not enough data to form a batchsize dataset
                if i == train_loader.dataset.__len__() // self.batch_size:
                    break
                real_labels = torch.ones(self.batch_size)
                fake_labels = torch.zeros(self.batch_size)
                images = Variable(images).cuda(self.cuda_index)
                real_labels, fake_labels = Variable(real_labels).cuda(self.cuda_index), Variable(fake_labels).cuda(
                    self.cuda_index)
                random_sampled_z = Variable(torch.randn(self.batch_size, latent_dim)).cuda(self.cuda_index)

                latent_code_z, z_mean, z_logvar = self.E(images)
                decoded_images = self.D(latent_code_z)
                randn_z_decode_images = self.D(random_sampled_z)

                score_true_data, _ = self.Dis(images)
                score_randn_gen, _ = self.Dis(randn_z_decode_images)
                score_latentz_gen, _ = self.Dis(decoded_images)
                loss1 = self.BCEloss(score_true_data.flatten(), real_labels)
                loss2 = self.BCEloss(score_randn_gen.flatten(), fake_labels)
                loss3 = self.BCEloss(score_latentz_gen.flatten(), fake_labels)
                discriminator_loss = loss1 + loss2 + loss3  # loss to be used to update discriminator
                self.Dis.zero_grad()
                discriminator_loss.backward(retain_graph=True)
                self.dis_optimizer.step()

                score_randn_gen, _ = self.Dis(randn_z_decode_images)
                score_latentz_gen, _ = self.Dis(decoded_images)
                loss2 = self.BCEloss(score_randn_gen.flatten(), real_labels)
                loss3 = self.BCEloss(score_latentz_gen.flatten(), real_labels)
                # discriminator_loss = loss1 + loss2 + loss3
                _, recon_Distibution = self.Dis(decoded_images)
                _, orig_Distribution = self.Dis(images)
                # _, recon_Distibution2 = self.Dis(randn_z_decode_images)
                recon_loss = (0.5 * (recon_Distibution - orig_Distribution) ** 2).sum()
                # recon_loss2 = (0.5 * (recon_Distibution2 - orig_Distribution) ** 2).sum()
                decoder_loss = recon_loss +loss2+loss3
                self.D.zero_grad()
                decoder_loss.backward(retain_graph=True)
                self.d_optimizer.step()
                # Optimize Encoder, Decoder and discriminator
                latent_code_z, z_mean, z_logvar = self.E(images)
                decoded_images = self.D(latent_code_z)
                recon_score, recon_Distibution = self.Dis(decoded_images)
                _, orig_Distribution = self.Dis(images)
                recon_loss = (0.5 * (recon_Distibution - orig_Distribution) ** 2).sum()
                recon_score_loss = self.BCEloss(recon_score.flatten(),real_labels)
                KLdiv = self.loss_function(z_mean, z_logvar)
                KLdiv /= images.shape[0]
                encoder_loss = KLdiv + recon_loss + recon_score_loss
                self.E.zero_grad();
                encoder_loss.backward(retain_graph=True)
                self.e_optimizer.step();
                generator_iter += 1
                if generator_iter % 300 == 0:
                    ## inception score###########################
                    sample_list = []
                    for i in range(10):
                        z = Variable(torch.randn(10, latent_dim)).cuda(self.cuda_index)
                        samples = self.D(z)
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
                    z = Variable(torch.randn(self.batch_size, latent_dim)).cuda(self.cuda_index)
                    samples = self.D(z)
                    samples = samples.mul(0.5).add(0.5)
                    samples = samples.data.cpu()[:self.batch_size]
                    grid = utils.make_grid(samples)
                    # utils.save_image(grid, 'vaegan128newh2z/randn_z_iter_{}.png'.format(str(generator_iter).zfill(3)))
                    utils.save_image(grid, 'vaegan128final/randn_z_iter_{}.png'.format(str(generator_iter).zfill(3)))
                    # Denormalize images and save them in grid 8x8
                    images_to_reconstruct = images[:self.batch_size]
                    z, _, _ = self.E(images_to_reconstruct)
                    samples = self.D(z)
                    samples = samples.mul(0.5).add(0.5)
                    samples = samples.data.cpu()[:self.batch_size]
                    grid = utils.make_grid(samples)
                    # utils.save_image(grid, 'vaegan128newh2z/sampled_z_iter_{}.png'.format(str(generator_iter).zfill(3)))
                    utils.save_image(grid, 'vaegan128final/sampled_z_iter_{}.png'.format(str(generator_iter).zfill(3)))
                    time = t.time() - self.t_begin
                    #print("Inception score: {}".format(inception_score))
                    print("Generator iter: {}".format(generator_iter))
                    print("Time {}".format(time))
                    dict = {'iters':[generator_iter],'encloss':[encoder_loss.cpu().data.numpy()],'decloss':[decoder_loss.cpu().data.numpy()],'disloss':[discriminator_loss.cpu().data.numpy()],'inscore0': [inception_score[0]],'inscore1': [inception_score[1]]}
                    df=pd.DataFrame(dict)
                    df.to_csv('gallery/vaegan128final_loss_inscore.csv', mode='a', index=False, header=False)
                    # Write to file inception_score, gen_iters, time
                    #output = str(generator_iter) + " " + str(time) + " " + str(inception_score[0]) + "\n"
                    #self.file.write(output)


                if ((i + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] Encoder_loss: %.8f, Decoder_loss: %.8f, Discriminator_loss: %.8f" %
                          ((epoch + 1), (i + 1), train_loader.dataset.__len__() // self.batch_size, encoder_loss.data, decoder_loss.data, discriminator_loss.data))
            self.lr_encoder.step()
            self.lr_decoder.step()
            self.lr_discriminator.step()

        self.t_end = t.time()
        print('Time of training-{}'.format((self.t_end - self.t_begin)))
        # Save the trained parameters
        self.save_model()

    def predict_label(self, train_loader):
        self.load_model()

        for k in range(40): # k is the number of certain attribute
            if k==1:
                break
            # initialize SVM model
            model = SVC()
            # define feature and label vectors
            feature_vectors = [] # n*128
            labels_list = [] # n*1
            z_new = None
            # loop over images
            for i, (images, labels) in enumerate(train_loader):
                if self.cuda:
                    images = Variable(images).cuda(self.cuda_index)
                z, _, _ = self.E(images)
                for j in range(self.batch_size):
                    print('i,j:', i, j)
                    feature_vectors.append(z[j].cpu().detach().numpy())
                    labels_list.append(labels[j][k].cpu().numpy())
                if i==10:
                    z_new = z
                    break
            print(feature_vectors)
            X = np.array(feature_vectors)
            y = np.array(labels_list)
            print(X)
            print('Y',y)
            model.fit(X, y)
            predicted_labels = model.predict(z_new.cpu().detach().numpy())
            print(predicted_labels)
        #
        # # get attribute vectors for each latent code
        # attribute_vectors = []
        # for label in predicted_labels:
        #     # convert label to binary attribute vector
        #     attribute_vector = [0 if x == -1 else 1 for x in label]
        #     # add to list of attribute vectors
        #     attribute_vectors.append(attribute_vector)


    def load_model(self):
        encoder_path = os.path.join('VAEGAN128final celebA128 30 16','encoder.pkl')
        decoder_path = os.path.join('VAEGAN128final celebA128 30 16', 'decoder.pkl')
        discriminator_path=os.path.join('VAEGAN128final celebA128 30 16', 'discriminator.pkl')
        # encoder_path = os.path.join('VAEGAN128new horse2zebra 6000 32','encoder.pkl')
        # decoder_path = os.path.join('VAEGAN128new horse2zebra 6000 32', 'decoder.pkl')
        # discriminator_path=os.path.join('VAEGAN128new horse2zebra 6000 32', 'discriminator.pkl')
        self.E.load_state_dict(torch.load(encoder_path))
        self.D.load_state_dict(torch.load(decoder_path))
        self.Dis.load_state_dict(torch.load(discriminator_path))
    def evaluate(self, test_loader, E_model_path , D_model_path, Dis_model_path ):
        self.load_model(E_model_path , D_model_path, Dis_model_path)
        # samples = torch.tensor()
        for i ,(images, _) in enumerate(test_loader):
            images=images[:64]
            z=self.E(images)
            sample=self.D(z)
            sample=sample.mul(0.5).add(0.5)
            sample = sample.data.cpu()
            grid = utils.make_grid(sample)
            print("Grid of 8x8 images saved to 'vaegan_model_image.png'.")
            utils.save_image(grid, 'vaegan_model_image.png')
            break


    def real_images(self, images, number_of_images):
        if (self.C == 3):
            return self.to_np(images.view(-1, self.C, celebA_size , celebA_size)[:self.number_of_images])
        else:
            return self.to_np(images.view(-1, celebA_size, celebA_size)[:self.number_of_images])

    def generate_img(self, z, number_of_images):
        samples = self.D(z).data.cpu().numpy()[:number_of_images]
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
        torch.save(self.E.state_dict(), './encoder.pkl')
        torch.save(self.D.state_dict(), './decoder.pkl')
        torch.save(self.Dis.state_dict(),'./discriminator.pkl')
        os.chdir('..')
        print('Models save to ./encoder.pkl & ./decoder.pkl & ./discriminator.pkl ')
    def adain_test(self,two_images, iter):
        self.load_model()
        if self.cuda:
            two_images=two_images.cuda()
        z,_,_=self.E(two_images)
        z1=z[0]
        z2=z[1]
        mean1=torch.sum(z1)/latent_dim
        mean2=torch.sum(z2)/latent_dim
        var1=torch.sum( (z1-mean1)**2 )/(latent_dim-1)
        var2 = torch.sum((z2 - mean2) ** 2) / (latent_dim - 1)
        number_int = 10
        z3 =var2*(z1-mean1)/var1 + mean2
        z4 = var1 * (z2 - mean2) / var2 + mean1
        print(z3)
        if self.cuda:
            z3 = z3.cuda()
            z4 = z4.cuda()
            z=torch.stack((z3,z4),dim=0)
        print(z.shape)
        print(z)
        two_generated = self.D(z)
        two_generated = two_generated.mul(0.5).add(0.5)
        images = []
        for i in range(2):
            images.append(two_images[i].view(3,128,128).data.cpu())
            images.append(two_generated[i].view( 3, 128, 128).data.cpu())
        grid = utils.make_grid(images, nrow=2)
        utils.save_image(grid, 'gallery/adain vaegan128/{}.png'.format(str(iter).zfill(3)))

    def generate_latent_walk(self,twenty_imgs):
        self.load_model()
        print('networks of E,D,loaded')
        # Interpolate between twe noise(z1, z2) with number_int steps between
        number_int = 10
        z_intp = torch.FloatTensor(number_int, latent_dim)
        z1 = torch.randn(number_int, latent_dim)
        z2 = torch.randn(number_int, latent_dim)
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
            fake_im = self.D(z_intp)
            fake_im = fake_im.mul(0.5).add(0.5) #denormalize
            for j in range(number_int):
                temp=fake_im[j]
                images.append(temp.view(self.C,celebA_size,celebA_size).data.cpu())
        grid = utils.make_grid(images,nrow=number_int)
        utils.save_image(grid, 'gallery/latent walk 128/vaegan new rw.png')
        print("Saved walk vaegan latent code/randn walk vaegan.png")
        ###################################################
        fourimgs=twenty_imgs[:4].cuda()
        z, _, _ = self.E(fourimgs)
        z1 = z[0:2]
        z2 = z[2:]
        z_intp = torch.FloatTensor(2, latent_dim)
        # if self.cuda:
        #     first_8_imgs = first_8_imgs.cuda()
        #     last_8_imgs = last_8_imgs.cuda()
        # z1, _, _ = self.E(first_8_imgs)
        # z2, _, _ = self.E(last_8_imgs)
        if self.cuda:
            z_intp = z_intp.cuda()
            z1 = z1.cuda()
            z2 = z2.cuda()
        z_intp = Variable(z_intp)
        images = []
        alpha = 1.0 / float(number_int + 1)
        print(alpha)
        for i in range(1, number_int + 1):
            z_intp.data = z1 * alpha + z2 * (1.0 - alpha)
            alpha += alpha
            fake_im = self.D(z_intp)
            fake_im = fake_im.mul(0.5).add(0.5)  # denormalize
            # for j in range(number_int):
            for j in range(1):
                temp = fake_im[j]
                images.append(temp.view(self.C, celebA_size, celebA_size).data.cpu())
        # grid = utils.make_grid(images, nrow=number_int)
        # utils.save_image(grid, 'gallery/2imgs walk vae128.png')
        # print("Saved walk vaegan latent code/2imgs walk vae128.png")
        return images

    def get_inception_score(self):
        self.load_model()
        sample_list = []
        for i in range(12):
            z = Variable(torch.randn(60, latent_dim)).cuda(self.cuda_index)
            samples = self.D(z)
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
        df.to_csv('gallery/vaegan128_loss_inscore_celebA.csv', mode='a', index=False, header=False)

    def get_codez(self, images):
        self.load_model()
        if self.cuda:
            images = images.cuda()
        z, _, _ = self.E(images)
        return z

    def compare_encoders(self, z1, z2):
        self.load_model()
        image2 = self.D(z2)
        image1 = self.D(z1)
        return image1,image2