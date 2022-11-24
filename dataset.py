from os.path import join

from torch.utils.data import Dataset
import torch

# TODO: Modify Dataset such that it gives out symbol images, coordinates, LOS
# graph also, along with current input (formula image )and output (label seq)
class Im2LatexDataset(Dataset):
    def __init__(self, data_dir, split, max_len):
        """args:
        data_dir: root dir storing the prepoccessed data
        split: train, validate or test
        """
        assert split in ["train", "validate", "test"]
        self.data_dir = data_dir
        self.split = split
        self.max_len = max_len
        self.input_tuples = self._load_input_tuples()

    def _load_input_tuples(self):
        input_tuples = torch.load(join(self.data_dir, "{}.pkl".format(self.split)))
        for i, (formula_imgs, coordinates, symbols, edge_indices, formula) in enumerate(input_tuples):
            input_tuple = (formula_imgs, coordinates, symbols, edge_indices, " ".join(formula.split()[:self.max_len]))
            input_tuples[i] = input_tuple
        return input_tuples

    def __getitem__(self, index):
        return self.input_tuples[index]

    def __len__(self):
        return len(self.input_tuples)
