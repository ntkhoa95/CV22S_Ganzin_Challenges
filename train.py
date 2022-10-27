import os, torch
import numpy as np
import stat, argparse
from tqdm import tqdm

from torch import Tensor
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from torch.autograd import Variable
from datasets.GANZIN_dataset import *
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from models import PUPNet

import warnings

from utils.util import *
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Training with PyTorch')
parser.add_argument('--dataset', type=str, default='ganzin', help='choosing dataset for training session')
parser.add_argument('--num_classes', type=int, default=2, help='number of classes in selected dataset')
parser.add_argument('--dataroot', type=str, default='./ganzin_dataset_final', help='directory of the loading data')
parser.add_argument('--resize_h', type=int, default=480, help='target resizing height')
parser.add_argument('--resize_w', type=int, default=640, help='target resizing width')

parser.add_argument('--num_resnet_layers', type=int, default=18, help='chooosing model for training session')
parser.add_argument('--model_name', type=str, default='PUPNet', help='chooosing model for training session')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='models are saved here')
parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs for training session')
parser.add_argument('--batch_size', type=int, default=8, help='number of images in a loading batch')
parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--gpu_ids', type=int, default=0, help='setting index of GPU for traing, "-1" for CPU')
parser.add_argument('--num_workers', type=int, default=8, help='number of workers for loading data')
parser.add_argument('--lr_decay', type=float, default=0.95, help='weight decay for adjusting learning rate')
parser.add_argument('--augmentation', type=bool, default=True, help='setting random augmentation')
parser.add_argument('--save_every', type=int, default=50, help='save model every defined epochs')
parser.add_argument('--visualization_flag', type=bool, default=False, help='setting flag for visualizing results during training session')

parser.add_argument('--verbose', type=bool, default=False, help='if specified, debugging size of each part of model')

args = parser.parse_args()

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

def train(epoch, model, train_loader, optimizer):
    model.train()
    for it, (imgs, labels, names) in tqdm(enumerate(train_loader)):
        imgs = Variable(imgs).cuda(args.gpu_ids)
        labels = Variable(labels).cuda(args.gpu_ids)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = F.cross_entropy(logits, labels) + dice_loss(F.softmax(logits, dim=1).float(),
                                       F.one_hot(labels, args.num_classes).permute(0, 3, 1, 2).float(), multiclass=True)
        loss.backward()
        optimizer.step()

        if args.visualization_flag:
            visualise(image_names=names, imgs=imgs, labels=labels, predictions=logits.argmax(1), dataset_name='ganzin', phase='train')

        if acc_iter['train'] % 1 == 0:
            writer.add_scalar("Train/loss", loss, acc_iter['train'])
        acc_iter['train'] += 1

def validation(epoch, model, val_loader):
    model.eval()
    iou_meter = AverageMeter()
    iou_meter_sequence = AverageMeter()
    label_validity = []
    output_conf = []
    with torch.no_grad():
        for it, (imgs, labels, names) in tqdm(enumerate(val_loader)):
            imgs = Variable(imgs).cuda(args.gpu_ids)
            labels = Variable(labels).cuda(args.gpu_ids)
            logits = model(imgs)
            label = labels.cpu().detach().numpy().squeeze()
            output = logits.argmax(1).cpu().detach().numpy().squeeze()

            # Get mean confidence
            probs = F.softmax(logits, dim=1).cpu().detach().numpy().squeeze()
            probs = np.max(probs, axis=0)
            conf = np.mean(probs)
            
            if np.sum(output.flatten()) > 50:
                conf = conf
            else:
                conf = 1 - conf

            # Setting color
            label_0 = [0, 0, 0]
            label_1 = [255, 0, 255]
            palette = np.array([label_0, label_1]).tolist()

            # Blending color to prediction
            pred_img  = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)
            for cid in range(2):
                pred_img[output == cid] = palette[cid]

            # Blending color to ground-truth
            label_img  = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
            for cid in range(2):
                label_img[label == cid] = palette[cid]

            if args.visualization_flag:
                visualise(image_names=names, imgs=imgs, labels=labels, predictions=logits.argmax(1), dataset_name='ganzin', phase='val')

            if np.sum(label_img.flatten()) > 0:
                label_validity.append(1.0)
                iou = mask_iou(pred_img, label_img)
                iou_meter.update(conf * iou)
                iou_meter_sequence.update(conf * iou)
            else:  # empty ground truth label
                label_validity.append(0.0)
            output_conf.append(conf)

    tn_rates = true_negative_curve(np.array(output_conf), np.array(label_validity))
    wiou = iou_meter.avg()
    atnr = np.mean(tn_rates)
    score = 0.7 * wiou + 0.3 * atnr
    print(f'\n\nOverall weighted IoU: {wiou:.4f}')
    print(f'Average true negative rate: {atnr:.4f}')
    print(f'Evaluated score: {score:.4f}')

    return score


def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    for it, (image, mask_true, names) in tqdm(enumerate(val_loader)):
#     for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
#         image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, args.num_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if args.num_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), args.num_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches

if __name__ == "__main__":
    torch.cuda.set_device(args.gpu_ids)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = eval(args.model_name)(n_class=args.num_classes, num_resnet_layers=args.num_resnet_layers, verbose=args.verbose)
    if args.gpu_ids >= 0: model.cuda(args.gpu_ids)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)  # goal: maximize Dice score

    # Prepare folder
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    experiment_ckpt_dir = os.path.join(args.checkpoint_dir)
    os.makedirs(experiment_ckpt_dir, exist_ok=True)

    # Creating writter to save training logs
    writer = SummaryWriter(f"{experiment_ckpt_dir}/tensorboard_log")

    print("Training on GPU: ", args.gpu_ids)
    print("Training log saved to: ", experiment_ckpt_dir)
    print(f"Using ResNet-{args.num_resnet_layers} backbone")

    # Setting datasets
    train_dataset = GANZIN_dataset(data_path=args.dataroot, phase='train', transform=True)
    val_dataset   = GANZIN_dataset(data_path=args.dataroot, phase='val', transform=False)

    train_loader  = DataLoader(dataset=train_dataset, \
                                batch_size=args.batch_size,
                                    shuffle=True,
                                        num_workers=args.num_workers,
                                            pin_memory=True,
                                                drop_last=False)
    val_loader    = DataLoader(dataset=val_dataset, \
                                batch_size=1,
                                    shuffle=False,
                                        num_workers=args.num_workers,
                                            pin_memory=True,
                                                drop_last=False)

    best_score = 0
    acc_iter = {"train": 0, "val": 0}
    for epoch in range(1, args.num_epochs+1):
        print(f"\nTraining {args.model_name} | Epoch {epoch}/{args.num_epochs}")
        train(epoch, model, train_loader, optimizer)
#         score = validation(epoch, model, val_loader)
        score = evaluate(model, val_loader, device)
        print('Validation Dice :', score)
        checkpoint_model_file = os.path.join(experiment_ckpt_dir, 'latest_model.pth')
        print('Saving latest checkpoint model!')
        torch.save(model.state_dict(), checkpoint_model_file)

        if epoch % args.save_every == 0:
            checkpoint_model_file = os.path.join(experiment_ckpt_dir, str(epoch)+'_model.pth')
            print('Saving checkpoint model!')
            torch.save(model.state_dict(), checkpoint_model_file)

        if epoch == 1:
            best_score = score
            checkpoint_pre_model_file = os.path.join(experiment_ckpt_dir, 'best_model.pth')
            torch.save(model.state_dict(), checkpoint_pre_model_file)
        else:
            if score > best_score:
                best_score = score
                print('Saving best checkpoint model!')
                checkpoint_pre_model_file = os.path.join(experiment_ckpt_dir, 'best_model.pth')
                torch.save(model.state_dict(), checkpoint_pre_model_file)
        scheduler.step(score)