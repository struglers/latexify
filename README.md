# Latexify
A machine learning model to convert handwritten text to LaTeX code.

## Usage

### Dependencies
Install miniconda from [here](https://docs.conda.io/en/latest/miniconda.html "Miniconda Installation Page"). Create a conda virtual environment and install necessary dependencies using the following command. Activate the virtual environment to use installed packages.

``` sh
conda env create -f environment.yml
conda activate latexify-env
```

### Dataset
Download the dataset from [IM2LATEX-100K dataset from Kaggle](https://www.kaggle.com/datasets/shahrukhkhan/im2latex100k) and extract it in the `./data` folder. Check if all the images are present in the `./data/formula_images_processed/formula_images_processed` folder.

### Preprocess
Run the following to preprocess data and save relevant `pkl` files that will be used for training and evaluation later on.

``` sh
python preprocess.py
```

### Build Vocabulary
Run the following to create a `vocab.pkl` file. This contains mappings from latex tokens to token ids.
``` sh
python build_vocab.py
```

### Training
You can use the `train.py` script to train the model. Run the following to see all the argument you can provide to the script.

``` sh
python train.py --help
```

An example looks like the following. This assumes that the data is present in the `./data` folder and saves the trained models to the `./checkpoints` folder.

``` sh
python train.py \
     --data_path=data \
     --save_dir=checkpoints \
     --dropout=0.2 --add_position_features \
     --epochs=25 --max_len=150
```

### Evaluation
Use the `eval.py` script to evaluate the model. An example looks like the following. This saves the predicted token sequences in the `./results/eval.txt` directory and their corresponding actual tokens in the `./refs/ref.txt` directory.

``` sh
python evaluate.py --split=test \
     --model_path=checkpoints/best_ckpt.pt \
     --data_path=data \
     --batch_size=32 \
     --ref_path=refs/ref.txt \
     --result_path=results/eval.txt \
     --beam_size=1
```
