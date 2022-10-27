import numpy as np
from PIL import Image
import os, torch, cv2
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

def get_palette(dataset="ganzin"):
    """Visualizing segmentation results in colormap"""
    assert dataset in ["gmrpd", "cityscapes", "thermal", "ganzin"]
    if dataset == "ganzin":
        label_0 = [0, 0, 0]
        label_1 = [255, 0, 255]
        palette = np.array([label_0, label_1]).tolist()
    return palette

def visualise(image_names, imgs, labels, predictions, dataset_name="ganzin", phase='train'):
    # print(imgs.shape, labels.shape, predictions.shape)
    palette = get_palette(dataset=dataset_name)
    os.makedirs(f'./checkpoints/visualization/{phase}', exist_ok=True)
    if phase == 'train':
        img_name = image_names[-1].split(".")[0]

        input_rgb   = imgs[-1].cpu().numpy().transpose(1, 2, 0) * 255

        input_rgb = Image.fromarray(np.uint8(input_rgb))
        input_rgb.save((f'./checkpoints/visualization/{phase}/{img_name}_rgb.png'))

        input_label = labels[-1].cpu().numpy()
        label_img   = np.zeros((input_label.shape[0], input_label.shape[1], 3), dtype=np.uint8)
        for cid in range(len(palette)):
            label_img[input_label == cid] = palette[cid]
        label_img = Image.fromarray(np.uint8(label_img))
        label_img.save(f'./checkpoints/visualization/{phase}/{img_name}_label.png')

        pred = predictions[-1].cpu().numpy()
        pred_img  = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for cid in range(len(palette)):
            pred_img[pred == cid] = palette[cid]
        pred_img = Image.fromarray(np.uint8(pred_img))
        pred_img.save(f'./checkpoints/visualization/{phase}/{img_name}_pred.png')
    else:
        for (idx, pred) in enumerate(predictions):
            img_name = image_names[idx].split(".")[0]

            input_rgb   = imgs[idx].cpu().numpy().transpose(1, 2, 0) * 255

            input_rgb = Image.fromarray(np.uint8(input_rgb))
            input_rgb.save((f'./checkpoints/visualization/{phase}/{img_name}_rgb.png'))

            input_label = labels[idx].cpu().numpy()
            label_img   = np.zeros((input_label.shape[0], input_label.shape[1], 3), dtype=np.uint8)
            for cid in range(len(palette)):
                label_img[input_label == cid] = palette[cid]
            label_img = Image.fromarray(np.uint8(label_img))
            label_img.save(f'./checkpoints/visualization/{phase}/{img_name}_label.png')

            pred = predictions[idx]
            pred_img  = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
            for cid in range(len(palette)):
                pred_img[pred == cid] = palette[cid]
            pred_img = Image.fromarray(np.uint8(pred_img))
            pred_img.save(f'./checkpoints/visualization/{phase}/{img_name}_pred.png')

# helper loading function that can be used by subclasses
def load_network(network, loading_epoch, save_dir=''):
    save_filename = '%s_model.pth' % (loading_epoch)
    save_path = os.path.join(save_dir, save_filename)
    print(save_path)
    if not os.path.isfile(save_path):
        print('%s not exists yet!' % save_path)
    else:
        #network.load_state_dict(torch.load(save_path))
        try:
            # print torch.load(save_path).keys()
            # print network.state_dict()['Scale.features.conv2_1_depthconvweight']
            network.load_state_dict(torch.load(save_path, map_location='cuda:0'))
        except:   
            pretrained_dict = torch.load(save_path, map_location='cuda:0')               
            model_dict = network.state_dict()
            try:
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}                    
                network.load_state_dict(pretrained_dict)
                print('Pretrained network has excessive layers; Only loading layers that are used' )
            except:
                print('Pretrained network has fewer layers; The following are not initialized:' )
                # from sets import Set
                # not_initialized = Set()
                for k, v in pretrained_dict.items():                      
                    if v.size() == model_dict[k].size():
                        model_dict[k] = v
                not_initialized=[]
                # print(pretrained_dict.keys())
                # print(model_dict.keys())
                for k, v in model_dict.items():
                    if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                        not_initialized+=[k]#[k.split('.')[0]]
                print(sorted(not_initialized))
                network.load_state_dict(model_dict)
    return network

def alpha_blend(input_image: np.ndarray, segmentation_mask: np.ndarray, alpha: float = 0.5):
    """Alpha Blending utility to overlay segmentation masks on input images
    Args:
        input_image: a np.ndarray with 1 or 3 channels
        segmentation_mask: a np.ndarray with 3 channels
        alpha: a float value
    """
    if len(input_image.shape) == 2:
        input_image = np.stack((input_image,) * 3, axis=-1)
    blended = input_image.astype(np.float32) * alpha + segmentation_mask.astype(np.float32) * (1 - alpha)
    blended = np.clip(blended, 0, 255)
    blended = blended.astype(np.uint8)
    return blended


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.sum = 0
        self.count = 0

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.sum += val
        self.count += 1

    def avg(self):
        return self.sum / self.count

def true_negative_curve(confs: np.ndarray, labels: np.ndarray, nr_thresholds: int = 1000):
    """Compute true negative rates
    Args:
        confs: the algorithm outputs
        labels: the ground truth labels
        nr_thresholds: number of splits for sliding thresholds

    Returns:

    """
    thresholds = np.linspace(0, 1, nr_thresholds)
    tn_rates = []
    for th in thresholds:
        # thresholding
        predict_negatives = (confs < th).astype(int)
        # true negative
        tn = np.sum((predict_negatives * (1 - labels) > 0).astype(int))
        tn_rates.append(tn / np.sum(1 - labels))
    return np.array(tn_rates)


def mask_iou(mask1: np.ndarray, mask2: np.ndarray):
    """Calculate the IoU score between two segmentation masks
    Args:
        mask1: 1st segmentation mask
        mask2: 2nd segmentation mask
    """
    if len(mask1.shape) == 3:
        mask1 = mask1.sum(axis=-1)
    if len(mask2.shape) == 3:
        mask2 = mask2.sum(axis=-1)
    area1 = cv2.countNonZero((mask1 > 0).astype(int))
    area2 = cv2.countNonZero((mask2 > 0).astype(int))
    if area1 == 0 or area2 == 0:
        return 0
    area_union = cv2.countNonZero(((mask1 + mask2) > 0).astype(int))
    area_inter = area1 + area2 - area_union
    return area_inter / area_union


def benchmark(dataset_path: str, subjects: list):
    """Compute the weighted IoU and average true negative rate
    Args:
        dataset_path: the dataset path
        subjects: a list of subject names

    Returns: benchmark score

    """
    iou_meter = AverageMeter()
    iou_meter_sequence = AverageMeter()
    label_validity = []
    output_conf = []
    sequence_idx = 0
    for subject in subjects:
        for action_number in range(26):
            image_folder = os.path.join(dataset_path, subject, f'{action_number + 1:02d}')
            sequence_idx += 1
            nr_image = len([name for name in os.listdir(image_folder) if name.endswith('.jpg')])
            iou_meter_sequence.reset()
            label_name = os.path.join(image_folder, '0.png')
            if not os.path.exists(label_name):
                print(f'Labels are not available for {image_folder}')
                continue
            for idx in tqdm(range(nr_image), desc=f'[{sequence_idx:03d}] {image_folder}'):
                image_name = os.path.join(image_folder, f'{idx}.jpg')
                label_name = os.path.join(image_folder, f'{idx}.png')
                image = cv2.imread(image_name)
                label = cv2.imread(label_name)
                # TODO: Modify the code below to run your method or load your results from disk
                # output, conf = my_awesome_algorithm(image)
                output = label
                conf = 1.0
                if np.sum(label.flatten()) > 0:
                    label_validity.append(1.0)
                    iou = mask_iou(output, label)
                    iou_meter.update(conf * iou)
                    iou_meter_sequence.update(conf * iou)
                else:  # empty ground truth label
                    label_validity.append(0.0)
                output_conf.append(conf)
            # print(f'[{sequence_idx:03d}] Weighted IoU: {iou_meter_sequence.avg()}')
    tn_rates = true_negative_curve(np.array(output_conf), np.array(label_validity))
    wiou = iou_meter.avg()
    atnr = np.mean(tn_rates)
    score = 0.7 * wiou + 0.3 * atnr
    print(f'\n\nOverall weighted IoU: {wiou:.4f}')
    print(f'Average true negative rate: {atnr:.4f}')
    print(f'Benchmark score: {score:.4f}')

    return score