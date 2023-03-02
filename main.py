from vae128 import Encoder as vaeenc
from vae128 import Decoder as vaedec
from vae128 import VAE_MODEL128
from dcgan128 import Generator as gangen
from dcgan128 import Discriminator as gandis
from vaegan128 import Encoder as vaegan1enc
from vaegan128 import Decoder as vaegan1dec
from vaegan128 import Discriminator as vaegan1dis
from vaegan128new import Encoder as vaegan2enc
from vaegan128new import Decoder as vaegan2dec
from vaegan128new import Discriminator as vaegan2dis
from vaegan128final import Encoder as vaegan3enc
from vaegan128final import Decoder as vaegan3dec
from vaegan128final import Discriminator as vaegan3dis
from torchsummary import summary
# model=VAE_MODEL128()
enc=vaegan2enc(3)
dec=vaegan2dec(3)
dis=vaegan2dis(3)
summary(enc,(3,128,128))
summary(dec,input_size=(128,))
summary(dis,input_size=(3,128,128))
# summary