import torch
from torch import nn
import torch.nn.functional as F

def conv_norm(in_channels, out_channels, kernel_size=3, act=True):

    layer = [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                padding=kernel_size//2, bias=False),
        nn.BatchNorm2d(out_channels)
    ]
    if act:
        layer += [nn.ReLU()]
    
    return nn.Sequential(*layer)

class DecoderBlock(nn.Module):
    '''Recebe a ativação do nível anterior do decoder `x_dec` e a ativação do 
    encoder `x_enc`. É assumido que `x_dec` possui uma resolução espacial
    menor que que `x_enc` e que `x_enc` possui número de canais diferente
    de `x_dec`.
    
    O módulo ajusta a resolução de `x_dec` para ser igual a `x_enc` e o número
    de canais de `x_enc` para ser igual a `x_dec`.'''

    def __init__(self, enc_channels, dec_channels):
        super().__init__()
        self.channel_adjust = conv_norm(enc_channels, dec_channels, kernel_size=1,
                                        act=False)
        self.mix = conv_norm(dec_channels, dec_channels)

    def forward(self, x_enc, x_dec):
        x_dec_int = F.interpolate(x_dec, size=x_enc.shape[-2:], mode="nearest")
        x_enc_ad = self.channel_adjust(x_enc)
        y = x_dec_int + x_enc_ad
        return self.mix(y)

class Decoder(nn.Module):

    def __init__(self, encoder_channels_list, decoder_channels):
        super().__init__()

        # Inverte lista para facilitar interpretação
        encoder_channels_list = encoder_channels_list[::-1]

        self.middle = conv_norm(encoder_channels_list[0], decoder_channels)
        blocks = []
        for channels in encoder_channels_list[1:]:
            blocks.append(DecoderBlock(channels, decoder_channels))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, features):

        # Inverte lista para facilitar interpretação
        features = features[::-1]

        x = self.middle(features[0])
        for idx in range(1, len(features)):
            # Temos um bloco a menos do que nro de features, por isso
            # o idx-1
            x = self.blocks[idx-1](features[idx], x)

        return x
