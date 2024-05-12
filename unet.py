import torch
import torch.nn as nn


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        c = self.conv(x)
        return self.pool(c), c


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, to_upsample, from_contract):
        upsampled = self.up(to_upsample)
        # upsampled is smaller than from_contract
        # we need to crop from_contract
        d_x = (from_contract.size(2) - upsampled.size(2)) // 2
        d_y = (from_contract.size(3) - upsampled.size(3)) // 2
        assert d_x >= 0 and d_y >= 0, "got : {}, {}".format(d_x, d_y)
        #  keep channels and batch size
        f_c_cropped = from_contract[:, :, d_x:-d_x, d_y:-d_y]
        # cat on channels dim
        catted = torch.cat([f_c_cropped, upsampled], dim=1)

        x = self.conv(catted)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down = nn.ModuleList(
            [
                DownBlock(in_channels, 64),  # 64 x 284 x 284
                DownBlock(64, 128),  # 128 x 140 x 140
                DownBlock(128, 256),  # 256 x 68 x 68
                DownBlock(256, 512),  # 512 x 32 x 32
            ]
        )
        self.middle = nn.Sequential(
            nn.Conv2d(512, 1024, 3),  # 1024 x 30 x 30
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3),  # 512 x 28 x 28
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.up = nn.ModuleList(
            [
                UpBlock(1024, 512),  # 512 x 56 x 56
                UpBlock(512, 256),  # 256 x 104 x 104
                UpBlock(256, 128),  # 128 x 200 x 200
                UpBlock(128, 64),  # 64 x 392 x 392
            ]
        )
        self.out = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        contract = []
        for down_block in self.down:
            x, bef_pool = down_block(x)
            contract.append(bef_pool)
        x = self.middle(x)
        assert (
            x.shape[1] == 1024 and x.shape[2] == 28 and x.shape[3] == 28
        ), "got : {}".format(x.shape)
        i = 0
        for up_block, contract_x in zip(self.up, contract[::-1]):
            x = up_block(x, contract_x)
            i += 1

        return self.out(x)


if __name__ == "__main__":
    model = UNet().cuda()
    x = torch.randn(1, 1, 572, 572).cuda()
    out = model(x)
    print(out.shape)
