# SC-VAMP for Superposition Code


## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)


## Introduction

Official implementation of spatial coupled VAMP algorithm from the paper:  
[**"Capacity-achieving sparse superposition codes with spatially coupled VAMP decoder"**](https://arxiv.org/abs/2504.13601)



## Installation

### Prerequisites
- Python >= 3.9 (only verified on Python=3.10)
- pip package manager

### Setup
```bash
# Clone repository
git clone https://github.com/yztfu/SC-VAMP-for-Superposition-Code.git
cd SC-VAMP-for-Superposition-Code

# recommended to use conda to create a virtual environment
conda create -n sc-vamp python=3.10
conda activate sc-vamp

# Install dependencies
pip install -r requirements.txt
```


## Usage

You can run a simple example using the following command. Feel free to customize the parameters in `test1.sh` as needed.

### Examples
```bash
# Run example test script
sh ./examples/test1.sh
```


## Citation

If you use this code in your research, please cite:
```bibtex
@misc{liu2025capacityachievingsparsesuperpositioncodes,
      title={Capacity-achieving sparse superposition codes with spatially coupled VAMP decoder}, 
      author={Yuhao Liu and Teng Fu and Jie Fan and Panpan Niu and Chaowen Deng and Zhongyi Huang},
      year={2025},
      eprint={2504.13601},
      archivePrefix={arXiv},
      primaryClass={cs.IT},
      url={https://arxiv.org/abs/2504.13601}, 
}
```

