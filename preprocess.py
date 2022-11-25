
from os.path import join
import argparse
import csv

import cv2
from torchvision import transforms
import torch
from tqdm import tqdm

from utils import extract_inputs_from_image


def preprocess(data_dir, split):
    assert split in ["train", "validate", "test"]

    print("Process {} dataset...".format(split))
    images_dir = join(data_dir, "formula_images_processed")
    split_file = join(data_dir, "im2latex_{}.csv".format(split))
    inputs = []
    transform = transforms.ToTensor()

    with open(split_file, 'r') as csvfile:
        f = csv.reader(csvfile)
        next(f, None) #Ignore heading
        for line in tqdm(f, desc=split):
            formula, img_name = line
            # load img and its corresponding formula
            img_path = join(images_dir, img_name)
            img = cv2.imread(img_path)

            # Converting 3-channel RGB to 1-channel grayscale (not binary)
            formula_img_tensor = torch.mean(transform(img), dim=0, keepdim=True)

            # TODO: check if the following code works as desired i.e. the input
            # tuple must have tensors of proper shape and value range. If not,
            # modify accordingly.
            try:
                coordinates, symbols, edge_indices = extract_inputs_from_image(img)
            except: #skip all samples that throw an error (desperate quickfix!)
                continue
            # coordinates, symbols, edge_indices = extract_inputs_from_image(img)

            coordinate_tensor = torch.tensor(coordinates)
            symbol_img_tensor = torch.tensor(symbols)
            los_graph_edge_indices_tensor = torch.tensor(edge_indices)

            input_tuple = (formula_img_tensor,
                           coordinate_tensor,
                           symbol_img_tensor,
                           los_graph_edge_indices_tensor,
                           formula)
            inputs.append(input_tuple)
        inputs.sort(key=img_size)

    out_file = join(data_dir, "{}.pkl".format(split))
    torch.save(inputs, out_file)
    print("Save {} dataset to {}".format(split, out_file))


def img_size(input_tuple: tuple) -> tuple:
    """Returns shape of image in input tuple as a tuple"""
    img, _, _, _, _ = input_tuple
    return tuple(img.size())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Im2Latex Data Preprocess Program")
    parser.add_argument("--data_path", type=str,
                        default="./data/", help="The dataset's dir")
    args = parser.parse_args()

    splits = ["validate", "test", "train"]
    for s in splits:
        preprocess(args.data_path, s)
