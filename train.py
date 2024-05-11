from oxford_pets_dataset import OxfordPetsDataset
from torch.utils.data import DataLoader

# import transforms
from torchvision.transforms import Compose, Resize
import torch.nn as nn
import torch
from unet import UNet

lossfn = nn.CrossEntropyLoss()
unet = UNet(in_channels=3, out_channels=1).cuda()


def main():
    # We need to transform it to be the same size
    dataset = OxfordPetsDataset(
        "data",
        mode="train",
        transform={
            "image": Compose([Resize((572, 572))]),
            "mask": Compose([Resize((388, 388))]),
        },
    )
    loader = DataLoader(dataset, batch_size=6, shuffle=True)
    optimizer = torch.optim.Adam(unet.parameters(), lr=1e-3)
    for i, (img, mask) in enumerate(loader):
        img, mask = img.cuda(), mask.cuda()
        out = unet(img)
        assert out.shape == mask.shape, f"got {out.shape}, expected {mask.shape}"
        loss = lossfn(out.view(out.shape[0], -1), mask.view(mask.shape[0], -1))

        print(f"Batch {i}, Loss: {loss.item()}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    main()
