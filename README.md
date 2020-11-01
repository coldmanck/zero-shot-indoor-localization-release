# Zero-Shot Indoor Localization
The official evaluation code of the paper **[Zero-Shot Multi-View Indoor Localization via Graph Location Networks](https://dl.acm.org/doi/10.1145/3394171.3413856)** which has been accepted at ACM MM 2020. This repo also includes two datasets (ICUBE & WCP) used in the paper and useful code snippets for reading datasets.

<img src="figs/intro.jpg" width="320"> <img src="figs/zero-shot-indoor-localization.jpg" width="800">

Please cite our paper if you use our code/datasets or feel inspired by our work :)
```
@inproceedings{chiou2020zero,
  title={Zero-Shot Multi-View Indoor Localization via Graph Location Networks},
  author={Chiou, Meng-Jiun and Liu, Zhenguang and Yin, Yifang and Liu, An-An and Zimmermann, Roger},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
  pages={3431--3440},
  year={2020}
}
```

Especially for ICUBE dataset, you might also want to cite the following paper:
```
@inproceedings{liu2017multiview,
  title={Multiview and multimodal pervasive indoor localization},
  author={Liu, Zhenguang and Cheng, Li and Liu, Anan and Zhang, Luming and He, Xiangnan and Zimmermann, Roger},
  booktitle={Proceedings of the 25th ACM international conference on Multimedia},
  pages={109--117},
  year={2017}
}
```

## Datasets
### Images
Note that for both datasets, the name of each image follows the naming of `{xx}_000{yy}.jpg` where `xx` is the location ID and `yy` in {`01`,`02`,...,`12`} means different views of the location and `01-04`, `05-08` and `09-12` are sets of four views of the location.
- [ICUBE Dataset](https://drive.google.com/drive/folders/1T0Dq8xuwL9myzVu_fZ4AylC2BJNRBNrg?usp=sharing): collected at the ICUBE building (21 Heng Mui Keng Terrace, Singapore 119613) at NUS. Put the image data under `data/icube/`. 
- [West Coast Plaza (WCP) Dataset](https://drive.google.com/drive/folders/1hFMAMnJPoUdnRVsCn6SQ6M5deC9h2wMj?usp=sharing): collected at the WCP shopping mall (154 West Coast Rd, Singapore 127371). Put the image data under `data/wcp/`.

### Annotations
Annotations are provided in this repository. We provide description for each file as follows:

- `{dataset}_path.txt`      -> Each line represents a "path" or "road", where for each line there are three numbers: (1) road ID (2) start location ID (3) end location ID
- `loc_vec.npy`             -> A 2D array in size of (n_location, 2) where the line *i* is the coordinate of location *i* in increasing order (the first line is location 1). Generated from `{dataset}_path.txt`.
- `adjacency_matrix.npy`    -> (*for zero-shot use*) Adjacency matrix of the locations. A 2D array in size of (n_location + 1, n_location + 1). The first row and column should be omitted before use. Generated from `loc_vec.npy`. Note that currently the `adjacency_matrix.npy` file for `wcp` dataset is missed; you have to generate it yourself according to `wcp_path.txt`.
- `nonzl_loc_to_dict.pkl` -> (*for zero-shot use*) A dictionary that maps dataset's **seen** location IDs into all (training) class IDs.
- `all_loc_to_dict.pkl` -> (*for zero-shot use*) A dictionary that maps all dataset's **seen & unseen** location IDs into all (training & validation) class IDs.

### Pre-trained Feature
- `loc_vec_trained_{214,394}.npy` -> (*for zero-shot use*) Trained node features with the proposed Map2Vec using `compute-loc_vec.py`.

## Installation

- Python 3.6 or higher
- PyTorch 1.1 (possibly compatible with versions from 0.4 to 1.5)
- Torchvision (to be installed along with PyTorch)
- Other packages in `requirements.txt` (including torch-geometric==1.3.0)
```
conda craete -n zsgln python=3.6 scipy numpy
conda activate zsgln
pip install -r requirements.txt
# And install PyTorch following official steps
```

## Checkpoints
Download the trained models [here](https://drive.google.com/drive/folders/18FOWTYAg502qc92UdTjNPKC59WngsP1R?usp=sharing) and put them into the `checkpoints` folder follow the following predefined structure. You can also run `./download_checkpoints.sh` to download them automatically (while you need to install `gdown` via `pip install gdown` first).

```
checkpoints
|-icube
  |-standard
    |-resnet152-best_model-gln.pth
    |-resnet152-best_model-gln_att.pth
  |-zero-shot
    |-resnet152-best_model-baseline.pth
    |-resnet152-best_model-gln.pth
    |-resnet152-best_model-gln_att.pth
|-wcp
  |-standard
    |-resnet152-best_model-gln.pth
    |-resnet152-best_model-gln_att.pth
  |-zero-shot
    |-resnet152-best_model-baseline.pth
    |-resnet152-best_model-gln.pth
    |-resnet152-best_model-gln_att.pth
```

## Evaluation
Default dataset is `icube`. You may change to `wcp` as needed. Note that the code assumes using 1 GPU & will take up around ~2GB memory. To change to `cpu` only, make some changes to the codes.

### Ordinary Indoor Localization
Note that the first number in printed `top1_count` corresponds to meter-level accuracy in Table 1 and the first 6 numbers correspond to the CDF curves in Figure 5.

#### GLN (GCN based)
```
python eval-gln.py --network gcn --dataset icube --ckpt checkpoints/icube/standard/resnet152-best_model-gln.pth
```

#### GLN + Attention (GAT based)
```
python eval-gln.py --network gat --dataset icube --ckpt checkpoints/icube/standard/resnet152-best_model-gln_att.pth
```

### Zero-shot Indoor Localization
Note that the printed `top1_count` corresponds to CDF@k and `Top{1,2,3,5,10} Acc` corresponds to Recall@k in Table 2. MED are manually computed with linear interpolation finding the value resulting in 0.5 in `top1_count`.
#### Baseline
```
python eval-zs_gln.py --network baseline --dataset icube --ckpt checkpoints/icube/zero-shot/resnet152-best_model-baseline.pth
```

#### GLN (GCN based)
```
python eval-zs_gln.py --network gcn --dataset icube --ckpt checkpoints/icube/zero-shot/resnet152-best_model-gln.pth
```

#### GLN + Attention (GAT based)
```
python eval-zs_gln.py --network gat --dataset icube --ckpt checkpoints/icube/zero-shot/resnet152-best_model-gln_att.pth
```

### (Optional) Computing Map2Vec 
Refer to `compute-loc_vec.py` for computing Map2Vec embeddings for both `icube` and `wcp` datasets. 

Note that currently the `adjacency_matrix.npy` file for `wcp` dataset is missed; you have to generate it yourself from `loc_vec.npy`. However, only for verifying/evaluation purpose, you may skip this step to use the `loc_vec_trained_394.npy` directly.

## License
Our code & datasets are released under the MIT license. See [LICENSE](LICENSE) for additional details.

## Enquiry
Feel free to drop an email to mengjiun.chiou@u.nus.edu
