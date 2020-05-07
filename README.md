# NHNE

The implementation for "[Nonuniform Hyper-Network Embedding with Dual Mechanism](https://dl.acm.org/doi/10.1145/3388924)" (TOIS)



### Requirements

```
pip install -r requirements.txt
```



### Basic Usage

```
python src/main.py --input graph/<dataset>/<edgelist>
```



### Options

You can check out the other options available using:

```
python src/main.py --help
```



### Citation

```
@article{huang2020nonuniform,
  title={Nonuniform Hyper-Network Embedding with Dual Mechanism},
  author={Huang, Jie and Chen, Chuan and Ye, Fanghua and Hu, Weibo and Zheng, Zibin},
  journal={ACM Transactions on Information Systems (TOIS)},
  volume={38},
  number={3},
  pages={1--18},
  year={2020},
  publisher={ACM New York, NY, USA}
}

@inproceedings{huang2019hyper2vec,
  title={Hyper2vec: Biased Random Walk for Hyper-network Embedding},
  author={Huang, Jie and Chen, Chuan and Ye, Fanghua and Wu, Jiajing and Zheng, Zibin and Ling, Guohui},
  booktitle={DASFAA},
  year={2019}
}
```
