# Latexify
A machine learning model to convert handwritten text to LaTeX code.

# Dependencies
Install miniconda from [here](https://docs.conda.io/en/latest/miniconda.html "Miniconda Installation Page"). Create a conda virtual environment and install necessary dependencies using the following command. Activate the virtual environment to use installed packages.

``` sh
conda env create -f environment.yml
conda activate latexify-env
```

# Preprocessing
TODO: Create custom dataset with appropriate preprocessing pipeline
Preprocessing:
1. Remove whitespace and use cv2.findNonZero and zero padding to get an image of desired H and W. (Input 1 of shape 1*H*W)
   TODO: Decide on H and W values (hyperparameters)
2. Do symbol segmentation and get coordinates for each symbol. Resize each symbol and appropriately pad to get shape 1*32*32. (Input 2 of shape L*1*32*32)
   TODO: Integrate segmentation algorithm into the pipeline
3. Arrange coordinates in the order of node numbering (Input 3 of shape 4*L)
   TODO: Numbering must be done from left to right in an LOS fashion
4. Build an LOS graph using the segmented symbols. The expected output is the adjacency list tensor (dtype=long/int) (input 4 of shape (2, num_edges))
5. Get the latex formula from data and convert to tokens (using some token vocabulary)
   TODO: Download or get vocabulary
