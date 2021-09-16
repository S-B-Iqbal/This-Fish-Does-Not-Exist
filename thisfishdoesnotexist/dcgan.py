import torch.nn as nn

class DCGAN(nn.Module):
  """DCGAN Architecture
  NOTE: Generator & Discriminator are within the same class.
  TODO: Make Discriminator more modular.
  """

    def __init__(self, latent_dim=100,
                 num_feat_maps_gen=64,
                 num_feat_maps_dis=64,
                     color_channels=3):
        super().__init__()

        self.generator = nn.Sequential(
            nn.ConvTranspose2d(latent_dim,num_feat_maps_gen*8,
                               kernel_size=4,stride=1,padding=0,
                               bias=False),
            nn.BatchNorm2d(num_feat_maps_gen*8),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(),

            nn.ConvTranspose2d(num_feat_maps_gen*8, num_feat_maps_gen*4,
                               kernel_size=4,stride=2,padding=1,
                               bias=False),
            nn.BatchNorm2d(num_feat_maps_gen*4),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(),
            
            nn.ConvTranspose2d(num_feat_maps_gen*4,num_feat_maps_gen*2,
                               kernel_size=4,stride=2,padding=1,
                               bias=False),
            nn.BatchNorm2d(num_feat_maps_gen*2),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(),

            nn.ConvTranspose2d(num_feat_maps_gen*2, num_feat_maps_gen,
                               kernel_size=4,stride=2,padding=1,
                               bias=False),
            nn.BatchNorm2d(num_feat_maps_gen),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(),

            nn.ConvTranspose2d(num_feat_maps_gen, color_channels,
                               kernel_size=4,stride=2,padding=1,bias=False),
        nn.Tanh()

        )

        self.discriminator = nn.Sequential(
            nn.Conv2d(color_channels, num_feat_maps_dis,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(num_feat_maps_dis,num_feat_maps_dis*2 ,kernel_size=4,stride=2,padding=1, bias=False),
            nn.BatchNorm2d(num_feat_maps_dis*2),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(num_feat_maps_dis*2,num_feat_maps_dis*4 ,kernel_size=4,stride=2,padding=1, bias=False),
            nn.BatchNorm2d(num_feat_maps_dis*4),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(num_feat_maps_dis*4,num_feat_maps_dis*8 ,kernel_size=4,stride=2,padding=1, bias=False),
            nn.BatchNorm2d(num_feat_maps_dis*8),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(num_feat_maps_dis*8,1,kernel_size=4, stride=1, padding=0),
            nn.Flatten(),
        )

    def generator_forward(self,z):
        img = self.generator(z)
        return img
    def discriminator_forward(self,img):
        logits = model.discriminator(img)
        return logits
