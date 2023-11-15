import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score

from cloudmasking.models.registry import get_model
from cloudmasking.dataset.clouddataset import CloudDataset


def evaluate(args):

    output_dir = Path(args.output_dir) / args.exp_id
    save_dir = output_dir / "predicted_masks"
    checkpoint_dir = output_dir / "checkpoints"
    if not checkpoint_dir.exists():
        print(f"The experiment {args.exp_id} doesn't exists. Make sure it exists and it contains a checkpoint folder.")
        exit(0)

    save_dir.mkdir(parents=True, exist_ok=True)

    epoch = args.epoch
    model_path = checkpoint_dir / f"best_model_{epoch}.pth"
    if not model_path.exists():
        print(f"The checkpoint {model_path} does not exist.")
        exit(0)

    threshold = args.threshold

    model = get_model(args.model)
    model.load_state_dict(torch.load(str(model_path), map_location=torch.device('cpu')))

    dataset = CloudDataset(Path(args.val_dir))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    _ = model.eval()

    all_preds = list()
    all_true = list()
    for index, data in tqdm(enumerate(dataloader)):
        input_data, mask = data
        input_data = input_data.to(device)
        mask = mask.to(device)

        output = model(input_data)


        pred_mask = output > threshold
        if args.save_results:
            predicted_mask = torch.argmax(output, dim=1)
            predicted_mask = predicted_mask[0]
            predicted_mask = predicted_mask.cpu().numpy()
            predicted_mask = predicted_mask * 255
            predicted_mask = np.clip(predicted_mask, 0, 255).astype(np.uint8)
            predicted_mask = Image.fromarray(predicted_mask)
            predicted_mask.save(f"{save_dir}/{index}.png")

        pred_mask = pred_mask.view(-1).cpu().numpy()
        true_masks = mask.view(-1).cpu().numpy()

        all_preds.extend(pred_mask)
        all_true.extend(true_masks)

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    accuracy = accuracy_score(all_true, all_preds)
    precision = precision_score(all_true, all_preds)
    recall = recall_score(all_true, all_preds)
    f1 = 2 * precision * recall / (precision + recall)

    print("accuracy:", accuracy)
    print("precision:", precision)
    print("recall:", recall)
    print("f1:", f1)


def parse_args():
    """
    Command-line arguments to the system.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='Evaluation procedure')
    parser.add_argument('--exp_id', type=str, required=True, help='Id of the experiment. Must be unique')
    parser.add_argument('-od', '--output_dir', type=str, default='./output', help='path to store the output of training procedure.')
    parser.add_argument('-sr', '--save_results', action='store_true', help='Save predicted masks.')

    # Data args
    parser.add_argument('-vd', '--val_dir', type=str, help='Directory with valdation images and masks.')

    # Model args
    parser.add_argument('-m', '--model', type=str, default='unet', help='Model to use. Supported values are (`unet`).')
    parser.add_argument('-th','--threshold', type=float, default=0.5, help='Threhold for the probability to become of certain class. If prob > 0.5, then class = 1.')
    parser.add_argument('-ne', '--epoch', type=str, default="final", help='Epoch to load the weights from. It expects the file to exist.')


    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    print(args)

    evaluate(args)



if __name__ == '__main__':
    main()
