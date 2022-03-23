from .unet_part import *
import pdb
AE_channel_inter_medidate = 512

class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out #, attention


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y1 = self.avg_pool(x).view(b, c)
        y2 = self.max_pool(x).view(b, c)
        y = (y1 + y2) / 2
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ae_up_conv(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(ae_up_conv, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class Encoder(nn.Module):

    def __init__(self, latent_size, channel=3, mode='dae'):
        super().__init__()
        self.mode = mode
        # self.conv = nn.Sequential(
        #     nn.Conv2d(channel, 64, 5, stride=2, padding=2, dilation=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, 5, stride=2, padding=2, dilation=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 256, 5, stride=2, padding=2, dilation=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.Conv2d(256, AE_channel_inter_medidate, 5, stride=2, padding=2, dilation=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        # )
        if mode == 'svae':
            dimen = AE_channel_inter_medidate * 4 * latent_size
        elif mode == 'dvae':
            dimen = latent_size * 2
        elif mode == 'dae':
            dimen = latent_size
        else:
            raise ValueError('Wrong mode')
        self.inc = inconv(channel, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 256)
        self.down5 = down(256, 256)

        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * AE_channel_inter_medidate, dimen),
            nn.BatchNorm1d(dimen),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        if self.mode == 'dae':
            pass
        elif self.mode == 'svae' or self.mode == 'dvae':
            x6 = x6.view(x.size(0), -1)
            x6 = self.fc(x6)

        return x6

class SA_Encoder(nn.Module):

    def __init__(self, latent_size, channel=3, mode='dae'):
        super().__init__()
        self.mode = mode
        if mode == 'svae':
            dimen = AE_channel_inter_medidate * 4 * latent_size
        elif mode == 'dvae':
            dimen = latent_size * 2
        elif mode == 'dae':
            dimen = latent_size
        else:
            raise ValueError('Wrong mode')
        self.inc = inconv(channel, 32)
        self.down1 = down(32, 64)
        self.atten1 = Self_Attn(64, 'relu')
        self.down2 = down(64, 128)
        self.atten2 = Self_Attn(128, 'relu')
        self.down3 = down(128, 256)
        self.atten3 = Self_Attn(256, 'relu')
        self.down4 = down(256, 256)
        self.atten4 = Self_Attn(256, 'relu')
        self.down5 = down(256, 256)
        self.atten5 = SELayer(256)

        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * AE_channel_inter_medidate, dimen),
            nn.BatchNorm1d(dimen),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2_s = self.atten1(x2)
        x3 = self.down2(x2_s)
        x3_s = self.atten2(x3)
        x4 = self.down3(x3_s)
        x4_s = self.atten3(x4)
        x5 = self.down4(x4_s)
        x5_s = self.atten4(x5)
        x6 = self.down5(x5_s)
        x6_s = self.atten5(x6)
        if self.mode == 'dae':
            pass
        elif self.mode == 'svae' or self.mode == 'dvae':
            x6_s = x6_s.view(x.size(0), -1)
            x6_s = self.fc(x6_s)

        return x6_s

class SA_Encoder2(nn.Module):

    def __init__(self, latent_size, channel=3, mode='dae'):
        super().__init__()
        self.mode = mode
        if mode == 'svae':
            dimen = AE_channel_inter_medidate * 4 * latent_size
        elif mode == 'dvae':
            dimen = latent_size * 2
        elif mode == 'dae':
            dimen = latent_size
        else:
            raise ValueError('Wrong mode')
        self.inc = inconv(channel, 32)
        self.down1 = down(32, 64)
        self.atten1 = Self_Attn(64, 'relu')
        self.down2 = down(64, 128)
        self.atten2 = Self_Attn(128, 'relu')
        self.down3 = down(128, 256)
        self.atten3 = Self_Attn(256, 'relu')
        self.down4 = down(256, 256)
        self.atten4 = Self_Attn(256, 'relu')
        self.down5 = down(256, 256)
        self.atten5 = SELayer(256)

        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * AE_channel_inter_medidate, dimen),
            nn.BatchNorm1d(dimen),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2_s = self.atten1(x2)
        x3 = self.down2(x2_s)
        x3_s = self.atten2(x3)
        x4 = self.down3(x3_s)
        x4_s = self.atten3(x4)
        x5 = self.down4(x4_s)
        x5_s = self.atten4(x5)
        x6 = self.down5(x5_s)
        x6_s = self.atten5(x6)
        if self.mode == 'dae':
            pass
        elif self.mode == 'svae' or self.mode == 'dvae':
            x6_s = x6_s.view(x.size(0), -1)
            x6_s = self.fc(x6_s)

        return x2_s, x4_s, x6_s


class Decoder(nn.Module):

    def __init__(self, latent_size, output_channel=3, mode='dae'):
        super().__init__()
        self.latent_size = latent_size
        self.mode = mode
        self.fc0 = nn.Sequential(
            nn.Linear(latent_size, 14 * 14 * AE_channel_inter_medidate),
            nn.BatchNorm1d(14 * 14 * AE_channel_inter_medidate),
            nn.ReLU()
        )
        # self.deconv = nn.Sequential(
        #     nn.ConvTranspose2d(AE_channel_inter_medidate, 256, 5, stride=2, padding=2, output_padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(256, 128, 5, stride=2, padding=2, output_padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64, 64, 5, stride=2, padding=2, output_padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, output_channel, 5, padding=2),
        #     nn.Sigmoid()
        # )
        self.up1 = ae_up_conv(256, 256)
        self.up2 = ae_up_conv(256, 256)
        self.up3 = ae_up_conv(256, 128)
        self.up4 = ae_up_conv(128, 64)
        self.up5 = ae_up_conv(64, 32)
        self.outc = outconv(32, output_channel)

    def forward(self, z):
        if self.mode == 'dvae':
            z = self.fc0(z)
            z = z.view(z.size(0), AE_channel_inter_medidate, 7, 7)
        elif self.mode == 'dae':
            pass
        x = self.up1(z)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        x = self.outc(x)

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, AE_channel_inter_medidate, 5, stride=2, padding=2),
            nn.BatchNorm2d(AE_channel_inter_medidate),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(14 * 14 * AE_channel_inter_medidate, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
