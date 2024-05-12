from oxford_pets_dataset import OxfordPetsDataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop
import torch.nn as nn
import torch
from unet import UNet
from tqdm import tqdm
from pathlib import Path
import os
from PIL import Image


def get_resize_transform(size, is_mask=False):
    return Resize(
        size, antialias=True, interpolation=Image.NEAREST if is_mask else Image.BILINEAR
    )


def main(args):
    EPOCHS = args.epochs
    DATA_DIR = args.data_dir
    CHECKPOINT_DIR = Path(args.checkpoint_dir)
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    SAVE_EACH_GLOBAL_STEP = args.save_each_global_step
    testloader = DataLoader(
        OxfordPetsDataset(
            DATA_DIR,
            mode="test",
            transform={
                "image": Compose([get_resize_transform((572, 572))]),
                "mask": Compose(
                    [get_resize_transform((572, 572), is_mask=True), CenterCrop(388)]
                ),
            },
        ),
        batch_size=6,
        shuffle=False,
    )

    def validate(model):
        model.eval()
        avg_loss = 0
        avg_dice = 0
        bar = tqdm(testloader, desc="Validation", total=len(testloader))
        step = 0
        with torch.inference_mode():
            for img, mask in bar:
                img, mask = img.cuda(), mask.cuda()
                with torch.autocast(
                    device_type="cuda" if torch.cuda.is_available() else "cpu"
                ):
                    out = model(img)
                    loss = lossfn(
                        out.view(out.shape[0], -1), mask.view(mask.shape[0], -1)
                    )
                    avg_loss += loss.item()
                    dice = dice_loss(out, mask)
                    avg_dice += dice.item()
                    step += 1
                    bar.set_postfix(avg_loss=avg_loss / step, avg_dice=avg_dice / step)
            model.train()
        return avg_loss / len(testloader), avg_dice / len(testloader)

    lossfn = nn.BCEWithLogitsLoss()

    def dice_loss(pred, target, smooth=1e-6):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

    unet = UNet(in_channels=3, out_channels=1).cuda()
    if args.load_checkpoint:
        try:
            unet.load_state_dict(torch.load(args.load_checkpoint))
            print("Checkpoint loaded: ", args.load_checkpoint)
        except Exception:
            print("Checkpoint not found")
    unet = torch.compile(unet, fullgraph=True)

    # We need to transform it to be the same size
    dataset = OxfordPetsDataset(
        DATA_DIR,
        mode="train",
        transform={
            "image": Compose([get_resize_transform((572, 572))]),
            "mask": Compose(
                [get_resize_transform((572, 572), is_mask=True), CenterCrop(388)]
            ),
        },
    )
    print("Length of dataset: ", len(dataset))
    loader = DataLoader(dataset, batch_size=3, shuffle=True)
    optimizer = torch.optim.Adam(unet.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=70, factor=0.7  # reduce by 30% on plateau
    )
    global_step = 0
    grad_scaler = torch.GradScaler()
    for epoch in range(EPOCHS):
        progress_bar = tqdm(
            enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}/{EPOCHS}"
        )
        for _, (img, mask) in progress_bar:
            img, mask = img.cuda(), mask.cuda()
            with torch.autocast(
                device_type="cuda" if torch.cuda.is_available() else "cpu"
            ):
                out = unet(img)
                bce_loss = lossfn(
                    out.view(out.shape[0], -1), mask.view(mask.shape[0], -1)
                )
                dice = dice_loss(out, mask)
                loss = bce_loss + dice

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            progress_bar.set_postfix(
                last_bce_loss=bce_loss.item(),
                global_step=global_step,
                lr=optimizer.param_groups[0]["lr"],
                last_dice_loss=dice.item(),
            )
            scheduler.step(loss)

            global_step += 1
            if global_step % SAVE_EACH_GLOBAL_STEP == 0:
                print("Saving checkpoint")
                # Since we use torch.compile, we need to save the original model
                torch.save(
                    unet._orig_mod.state_dict()
                    if hasattr(unet, "_orig_mod")
                    else unet.state_dict(),
                    CHECKPOINT_DIR / "unet.pth",
                )
        val_loss, dice = validate(unet)
        print(f"Validation loss: {val_loss}, Validation dice loss: {dice}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--save_each_global_step", type=int, default=100)
    parser.add_argument("--load_checkpoint", type=str, default="checkpoints/unet.pth")

    main(parser.parse_args())
