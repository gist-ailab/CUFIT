<p align="center">
   <h2 align="center">Curriculum Fine-tuning of Vision Foundation Model for Medical Image Classification Under Label Noise</h2>
   <p align="center">
      <a href="https://scholar.google.com/citations?user=Ctm3p8wAAAAJ&hl=en"><strong>Yeonguk Yu</strong></a>
      ·
      <a><strong>Minhwan Ko</strong></a>
      ·
      <a"><strong>Sungho Shin</strong></a>
      ·
      <a"><strong>Kangmin Kim</strong></a>
      ·
      <a"><strong>Kyoobin Lee</strong></a>
     <br>
     <a href="https://sites.google.com/view/gistailab/"><strong>Artificial Intelligence LAB</strong></a>
      GIST, South Korea
   </p>
   <h3 align="center">NeurIPS 2024 - Poster Presentation</h3>
</p>

<p align="center">
   <a href=''>
      <img src='https://img.shields.io/badge/CUFIT-Presentation-ffffff?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiIHdpZHRoPSIxOCIgaGVpZ2h0PSIxOCI+PHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPjxwYXRoIGQ9Ik0xOSAzSDVjLTEuMSAwLTIgLjktMiAydjE0YzAgMS4xLjkgMiAyIDJoMTRjMS4xIDAgMi0uOSAyLTJWNWMwLTEuMS0uOS0yLTItMnpNOSAxN0g3di01aDJ2NXptNCAwaC0ydi03aDJ2N3ptNCAwaC0yVjhoMnY5eiIvPjwvc3ZnPg==&logoColor=white&labelColor=black&color=9370DB' alt='Results'>
   </a>

   <a href="">
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=for-the-badge&logo=adobeacrobatreader&logoWidth=20&logoColor=white&labelColor=black&color=94DD15' alt='Paper PDF'>
   </a>
</p>


Our **CU**rriculum **FI**ne-**T**uning of Vision Foundation Model **(CUFIT)** offers a robust training framework for medical multi-class image classification under noisy label conditions. 
Leveraging vision foundation models (VFMs) pretrained on large-scale datasets, CUFIT effectively handles noisy labels without modifying the feature extractor, using linear probing. Subsequently, it employs a curriculum fine-tuning approach, beginning with linear probing to ensure robustness to noisy samples, followed by fine-tuning two adapters for enhanced classification performance. CUFIT outperforms conventional methods across various medical image benchmarks, achieving superior results at various noise rates on datasets such as HAM10000 and APTOS-2019, highlighting its capability to address the challenges posed by noisy labels in medical datasets.

## 🚀 Getting Started
### Clone the Repository
   ```bash
   git clone https://github.com/gist-ailab/CUFIT.git
   cd CUFIT
   ```

### Environment Setup
   This code is tested under Linux 20.04 and Python 3.8.18 environment, and the code requires following main packages to be installed:
    
   - [Pytorch](https://pytorch.org/): Tested under 1.11.0 version of Pytorch-GPU.
   - [torchvision](https://pytorch.org/vision/stable/index.html): which will be installed along Pytorch. Tested under 0.6.0 version.
   - [timm](https://github.com/rwightman/pytorch-image-models): Tested under 0.4.12 version.
   - [scipy](https://www.scipy.org/): Tested under 1.4.1 version.
   - [scikit-learn](https://scikit-learn.org/stable/): Tested under 0.22.1 version.

you may use the follwoing lines.
```bash
conda env create -n cufit python=3.8
conda activate cufit
pip install -r requirement.txt
```


### Dataset Preparation
   Some public datasets are required to be downloaded for running evaluation.   
   - [HAM10000](https://challenge.isic-archive.com/data/#2018)
   - [APTOS-2019](https://www.kaggle.com/c/aptos2019-blindness-detection/data)

### Config file need to be changed for your path to download. For example,
~~~
# conf/ham10000.json
{
    "epoch" : "100",
    "id_dataset" : "./data/ham10000",   # Your path to dataset
    "batch_size" : 32,
    "save_path" : "./checkpoint/ham10000",   # Your path to checkpoint
    "num_classes" : 7
}
~~~


---
## How to Run
### To train a model by linear probing with DINOv2-small architecture
~~~
python train_linear.py -d 'data_name' -g 'gpu-num' -n resnet18 -s 'save_name'
~~~
for example,
~~~
python train_baseline.py -d cifar10 -g 0 -n resnet18 -s baseline
~~~


### To train a model by CUFIT with DINOv2-small architecture
~~~
python eval.py -n resnet18 -d 'data_name' -g 'gpu_num' -s 'save_name' -m featurenorm
~~~
for example, 
~~~
python eval.py -n resnet18 -d cifar10 -g 0 -s baseline 
~~~

    
## 🤝 Acknowledgements & Support
This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. 2022-0-00951, Development of Uncertainty Aware Agents Learning by Asking Questions).

### 🌟 License
The source code of this repository is released only for academic use. See the [license](LICENSE) file for details.

### 📚 Citation
If you use CUFIT in your research, please consider citing us.
```bibtex
TBD
```
