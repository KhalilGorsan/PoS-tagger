# PoS-tagger
Part-Of-Speech tagger
-----------------------------------------------------------------------------
Install
-----------------------------------------------------------------------------
We use:
- [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
  to setup our environment,
- and python 3.8.3

To setup the environment:
```bash
conda --version

# Clone the repo and navigate to the project directory
git clone https://github.com/KhalilGorsan/PoS-tagger.git
cd PoS-tagger

# Create conda env
conda env create -f environment_cpu.yml
source activate pos-tagger

# Install pre-commit hooks
pre-commit install
