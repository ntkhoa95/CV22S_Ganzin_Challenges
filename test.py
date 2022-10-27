from distutils import text_file
import os, torch
import numpy as np
import stat, argparse
from tqdm import tqdm
from utils.util import *

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
parser.add_argument('--num_resnet_layers', type=int, default=18, help='chooosing model for training session')
parser.add_argument('--num_classes', type=int, default=2, help='number of classes in selected dataset')
parser.add_argument('--resize_h', type=int, default=480, help='target resizing height')
parser.add_argument('--resize_w', type=int, default=640, help='target resizing width')
parser.add_argument('--model_name', type=str, default='PUPNet', help='chooosing model for training session')
parser.add_argument('--load_epoch', type=str, default='best', help='chooosing model for eval session')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='models are saved here')
parser.add_argument('--gpu_ids', type=int, default=0, help='setting index of GPU for traing, "-1" for CPU')
parser.add_argument('--num_workers', type=int, default=6, help='number of workers for loading data')
parser.add_argument('--visualization_flag', type=bool, default=True, help='setting flag for visualizing results during training session')

parser.add_argument('--verbose', type=bool, default=False, help='if specified, debugging size of each part of model')

args = parser.parse_args()

torch.cuda.set_device(args.gpu_ids)

model = eval(args.model_name)(n_class=args.num_classes, num_resnet_layers=args.num_resnet_layers, verbose=args.verbose)
model_checkpoint_dir = os.path.join(args.checkpoint_dir)
load_network(model, args.load_epoch, model_checkpoint_dir)
model.eval()
if args.gpu_ids >= 0:
    model.cuda(args.gpu_ids)

def eval_my_model(frame):
    frame = np.transpose(frame, (2, 0, 1)) / 255.0
    frame = torch.FloatTensor(frame).unsqueeze(0).cuda(args.gpu_ids)
    logits = model(frame)
    probs = F.softmax(logits, dim=1).cpu().detach().numpy().squeeze()
    probs = np.max(probs, axis=0)
    pred = logits.argmax(1).cpu().detach().numpy().squeeze()
    conf = np.mean(probs)
    
#     k1 = 1/24
#     min_size = np.min(pred.shape)
#     def get_odd(x):
#         return int(2 * (x/2 + 1) - 1)
#     a1 = get_odd(k1 * min_size)
#     kernel_1 = np.ones((a1, a1), np.uint8)
#     pred = cv2.normalize(pred, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8UC1)
#     pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel_1)
    
    if np.sum(pred.flatten()) > 1000:
        conf = conf
    else:
        conf = 1 - conf
    label_0 = [0, 0, 0]
    label_1 = [255, 0, 255]
    
    palette = np.array([label_0, label_1]).tolist()
    pred_img  = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for cid in range(2):
        pred_img[pred == cid] = palette[cid]

    return pred_img, conf

def eval_test_set(dataset_path: str, subjects: list):
    """Compute the weighted IoU and average true negative rate
    Args:
        dataset_path: the dataset path
        subjects: a list of subject names

    Returns: benchmark score

    """
    sequence_idx = 0
    for subject in subjects:
        solution_folder = os.path.join(dataset_path, "output", f"resnet_{args.num_resnet_layers}", f"{subject}_solution")
        os.makedirs(solution_folder, exist_ok=True)
        for action_number in range(26):
            image_folder = os.path.join(dataset_path, subject, f'{action_number + 1:02d}')
            output_folder = os.path.join(solution_folder, f'{action_number + 1:02d}')
            os.makedirs(output_folder, exist_ok=True)
            # Write to conf.txt file
            txt_file = "conf.txt"
            file = open(os.path.join(solution_folder, f'{action_number + 1:02d}', txt_file),"w")

            sequence_idx += 1
            nr_image = len([name for name in os.listdir(image_folder) if name.endswith('.jpg')])
            for idx in tqdm(range(nr_image), desc=f'[{sequence_idx:03d}] {image_folder}'):
                image_path = os.path.join(image_folder, f'{idx}.jpg')
                label_name = f'{idx}.png'
                output_label_path = os.path.join(output_folder, label_name)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('float32')
                output, conf = eval_my_model(image)
                output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_label_path, output)
                file.write(str(round(conf, 2)))
                file.write("\n")
            file.close()

if __name__ == "__main__":
    dataset_path = './dataset/public'
    # subjects = ['S1', 'S2', 'S3', 'S4']
    subjects = ['S5']
    eval_test_set(dataset_path, subjects)