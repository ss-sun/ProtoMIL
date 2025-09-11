# ProtoMIL
Official implementation for MICCAI 2025 paper:
## Prototype-Based Multiple Instance Learning for Gigapixel Whole Slide Image Classification
[Arxiv Paper](https://arxiv.org/abs/2503.08384)

**Note**: For any questions regarding the paper and code, please contact Susu Sun (susu.sun@uni-tuebingen.de).


## ProtoMIL framework
The ProtoMIL approach consists of the following three stages.

<div style="displaystyle=block;align=center;"><p align="center" >
  <img src="framework.png"/ width="75%" height="75%">
  </p>
</div>

(a) Sparse concept discovery:
- Image features *x* are extracted from Whole Slide Image (WSI) patches using the histopathology vision-language model CONCH.
- A sparse autoencoder (SAE) (**E**, **D**) is trained to reconstruct the image features from the latent embedding *h*.
- By enforcing dim (*h*) > dim (*x* ), the SAE yields a sparse and highly interpretable latent embedding *h*.

(b) Interpretation and inspection:
- The probing image set is constructed by randomly sampling 10K normal patches and 10K tumor patches from the downstream task training set.
- For each non-zero neuron, the 10 patches with the highest activation values h_i are selected to build a prototype.
- A pathologist examines the selected patches and maps the prototypes to corresponding histopathology concepts and identifies spurious concepts.

(c) ProtoMIL training with human intervention on spurious concepts:
- For WSI classification, image features are first projected onto human-interpretable concept activation vectors *h*.
- Inherent interpretable ProtoMIL is trained to generate predictions as linear combinations of input concepts, and provide self-explanations.
- Users can perform interventions by perturbing the concept activation vectors.

## Installation
```
conda env create -f environment.yml
conda activate protomil
```
## Datasets
We perform experiments on the **Camelyon16** and **PANDA** datasets. Please find these datasets on their official websites. Please download both the WSI and the corresponding label mask in order to extract normal patches and tumor patches to create the probing image set.

- **Camelyon16** (https://camelyon16.grand-challenge.org/Data/)

- **PANDA** (https://panda.grand-challenge.org/data/)

