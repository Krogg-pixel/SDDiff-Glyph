## SDDiff-Glyph: Semantic-Driven Diffusion for Decorative Chinese Character Generation

This repository contains the official implementation of SDDiff-Glyph, a semantic-driven diffusion framework for generating decorative Chinese characters inspired by the traditional folk art Huaniaozi.

The method introduces structured semantic supervision into a latent diffusion model, enabling controllable generation of visually expressive glyph compositions that combine character structure with floral and avian decorative motifs.

This repository provides the training, inference, and evaluation code used in our experiments, together with example data and generated samples.

## Overview

Decorative Chinese character styles such as Huaniaozi represent a distinctive form of folk typography in which Chinese characters are intertwined with botanical and animal imagery.
Unlike standard font generation tasks that rely on strict glyph skeleton alignment, this form of visual design emphasizes semantic composition, decorative balance, and stylistic rhythm.

To address these characteristics, SDDiff-Glyph models the generation process through semantic-driven conditioning, allowing the diffusion model to learn stylistic structure from descriptive annotations rather than explicit geometric constraints.

The framework integrates semantic annotations with LoRA-based parameter-efficient fine-tuning, enabling stable training even with a relatively small dataset.

## Repository Structure

The repository is organized as follows:

```
SDDiff-Glyph
│
├── dataset
│   ├── analysis
│   │   └── semantic_elements_stats.py
│   ├── annotations
│   ├── samples
│   └── semantic_levels
│
├── training
│   └── train_text_to_image_lora.py
│
├── inference
│   └── infer_lora_batch.py
│
├── scripts
│   ├── train_struct.sh
│   └── infer.sh
│
├── evaluation
│   ├── compute_metrics.py
│   └── comparison_images
│
├── results
│   ├── ablation
│   ├── prompt_control
│   └── samples
│
├── weights
│   └── pytorch_lora_weights.safetensors
│
└── requirements.txt
```

## Installation

We recommend using Python 3.10 or later.

Install dependencies using:

pip install -r requirements.txt

The project mainly relies on the following libraries:

-   PyTorch
-   HuggingFace Diffusers
-   Transformers
-   Accelerate
-   PEFT
-   LPIPS
-   PyTorch-FID
-   scikit-image

## Dataset

The dataset used in this study consists of decorative Chinese character imagery compiled for research purposes and annotated with structured semantic descriptions.

The complete dataset cannot be publicly released due to copyright and data usage considerations. To avoid potential misuse, including unintended commercial use of the source materials, only a small subset of sample images and the annotation format are provided in this repository for demonstration purposes.

This repository therefore provides:

-   a small subset of **sample images**
-   **annotation formats**
-   **semantic metadata used for supervision**

These materials are sufficient to understand the dataset structure and reproduce the training pipeline.

Dataset directory:

```
dataset/
├── samples              example images
├── annotations          annotation files
└── semantic_levels      semantic supervision variants
```

The script

dataset/analysis/semantic_elements_stats.py

can be used to compute statistics of semantic elements in the annotated dataset.

Two annotation subsets are included in this repository:

- `annotations_625.jsonl`: the complete annotated dataset used in the full training experiments.
- `annotations_400.jsonl`: a partial semantic annotation subset used in the reduced-supervision experiments.

The naming convention reflects the approximate scale of the subset rather than the exact number of annotated samples.

## Training

The training process is based on the Stable Diffusion v1.5 backbone implemented in the HuggingFace Diffusers library. 

Training is performed using LoRA fine-tuning on a Stable Diffusion backbone.

Example command:

bash scripts/train_struct.sh

This script launches the training configuration used for the semantic-driven experiments.

The training process includes different levels of semantic supervision corresponding to the ablation study reported in the paper.

## Inference

After training, new decorative characters can be generated using:

bash scripts/infer.sh

or directly running

python inference/infer_lora_batch.py

The inference script supports batch generation and prompt-based semantic control.

## Evaluation

Quantitative evaluation is implemented in

evaluation/compute_metrics.py

The script computes several perceptual similarity metrics commonly used in image generation tasks:

-   FID
-   LPIPS
-   SSIM

Example usage:

python evaluation/compute_metrics.py

Example image sets used for evaluation are included for demonstration.

These include generated results from multiple methods as well as reference images.

Images from external generative platforms are included solely for qualitative comparison in the experimental analysis.

## Results

Example generation outputs are provided in

results/

The directory contains:

Ablation experiments

results/ablation

showing results under different semantic supervision levels.

Prompt-controlled generation

results/prompt_control

demonstrating how different semantic prompts influence decorative glyph generation.

Additional random samples

results/samples

illustrating the diversity of the generated outputs.

These examples correspond to representative results discussed in the paper.

## Model Weights

A pretrained LoRA checkpoint is included for demonstration:

weights/pytorch_lora_weights.safetensors

The provided LoRA checkpoint corresponds to the semantic supervision configuration used in the main experiment reported in the paper.

## Reproducibility

This repository includes all essential components required to reproduce the experiments:

-   training scripts
-   inference scripts
-   evaluation metrics
-   dataset annotation formats
-   example results

These components together form a complete experimental pipeline for semantic-driven decorative glyph generation.

## Citation

If you find this work useful in your research, please consider citing:

@article{SDDiffGlyph2026,
  title={SDDiff-Glyph: Semantic-Driven Diffusion for Decorative Chinese Character Generation},
  author={Qin, Jialin and Hu, Qingqing and Zhang, Hao},
  journal={The Visual Computer},
  year={2026}
}
## Acknowledgements

This project builds upon several open-source frameworks including:

-   HuggingFace Diffusers
-   PyTorch
-   PEFT

We thank the contributors of these projects for making their work publicly available.

## License

This project is released under the MIT License.
