
from os.path import join
import argparse
import csv

from PIL import Image
from torchvision import transforms
import torch


def preprocess(data_dir, split):
    assert split in ["train", "validate", "test"]

    print("Process {} dataset...".format(split))
    images_dir = join(data_dir, "formula_images_processed")
    split_file = join(data_dir, "im2latex_{}.csv".format(split))
    pairs = []
    transform = transforms.ToTensor()

    with open(split_file, 'r') as csvfile:
        f = csv.reader(csvfile)
        next(f, None) #Ignore heading
        for line in f:
            formula, img_name = line
            # load img and its corresponding formula
            img_path = join(images_dir, img_name)
            img = Image.open(img_path)
            # Converting RGB image to grayscale (not binary)
            img_tensor = transform(img).mean(dim=0, keepdims=True)
            pair = (img_tensor, formula)
            pairs.append(pair)
        pairs.sort(key=img_size)

    out_file = join(data_dir, "{}.pkl".format(split))
    torch.save(pairs, out_file)
    print("Save {} dataset to {}".format(split, out_file))


def img_size(pair):
    img, formula = pair
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
