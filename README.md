### Reproduce the experiments

1. Create a new environment with the following dependencies:
    - Pytorch : 
        ```bash 
        pip install torch torchvision torchaudio
        ```
    - Pytorch geometric, according to your Pytorch version and CUDA version: 
        ```bash 
        pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-{pytorch_version}+{cuda_version}.html
        ```
    - Seaborn:
        ```bash 
        pip install seaborn
        ```
    - Matplotlib:
        ```bash 
        pip install matplotlib
        ```
    - Pandas:
        ```bash
        pip install pandas
        ```
    - torch-conformal package, from the root of the repository, run:
        ```bash
        python setup.py install
        ```
2. Run a Jupyter kernel on the environment
3. Run the notebooks to reproduce the experiments