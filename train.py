from oxford_pets_dataset import OxfordPetsDataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize
import torch.nn as nn
import torch
from unet import UNet
from tqdm import tqdm
import logging
from pathlib import Path

EPOCHS = 10
DATA_DIR = "data"
CHECKPOINT_DIR = Path("checkpoints")
SAVE_EACH_GLOBAL_STEP = 100


def main():
    testloader = DataLoader(
        OxfordPetsDataset(
            "data",
            mode="test",
            transform={
                "image": Compose([Resize((572, 572))]),
                "mask": Compose([Resize((388, 388))]),
            },
        ),
        batch_size=6,
        shuffle=False,
    )

    def validate(model):
        model.eval()
        avg_loss = 0
        bar = tqdm(testloader, desc="Validation", total=len(testloader))
        step = 0
        with torch.inference_mode():
            for img, mask in bar:
                img, mask = img.cuda(), mask.cuda()
                out = model(img)
                loss = lossfn(out.view(out.shape[0], -1), mask.view(mask.shape[0], -1))
                avg_loss += loss.item()
                step += 1
                bar.set_postfix(avg_loss=avg_loss / step)
            model.train()
        return avg_loss / len(testloader)

    lossfn = nn.BCEWithLogitsLoss()
    unet = UNet(in_channels=3, out_channels=1).cuda()
    unet = torch.compile(unet, fullgraph=True)
    # We need to transform it to be the same size
    dataset = OxfordPetsDataset(
        "data",
        mode="train",
        transform={
            "image": Compose([Resize((572, 572))]),
            "mask": Compose([Resize((388, 388))]),
        },
    )
    print("Length of dataset: ", len(dataset))
    loader = DataLoader(dataset, batch_size=3, shuffle=True)
    optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)
    global_step = 0
    for epoch in range(EPOCHS):
        progress_bar = tqdm(
            enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}/{EPOCHS}"
        )
        for _, (img, mask) in progress_bar:
            img, mask = img.cuda(), mask.cuda()
            out = unet(img)
            loss = lossfn(out.view(out.shape[0], -1), mask.view(mask.shape[0], -1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(
                last_train_loss=loss.item(), global_step=global_step
            )

            global_step += 1
        val_loss = validate(unet)
        logging.info(f"Validation loss: {val_loss}")
        if global_step % SAVE_EACH_GLOBAL_STEP == 0:
            torch.save(unet.state_dict(), CHECKPOINT_DIR / f"unet_{global_step}.pth")


if __name__ == "__main__":
    main()
