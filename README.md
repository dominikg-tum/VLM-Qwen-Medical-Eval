# VLM-Seminar-Final-Submission
Master Seminar Submission for AI in Vision-Language Models in Medical Imaging.

## Overview of Contributions:

### Notebooks
**For the two [datasets](https://github.com/LijunRio/VLM-Seminar25-Dataset) ** — **(1) chest-xrays** and **(2) brain-mris** — there are **two / three respective notebooks** to solve the tasks of **(1) classification and grounding** and **(2) description, detection, and diagnosis** using the VLM **Qwen2.5-VL-72B-Instruct**. 

Results of the predicitons and evaluations are saved under `./results/"chest_xrays or nova_brain"/*.json`.

### Evaluation Metrics
The **evaluation metrics** in folder `./code/eval_scripts` are based on the [scripts from the dataset repo](https://github.com/LijunRio/VLM-Seminar25-Dataset).

### Prompts
The prompts used in the notebooks are based on the [suggestions](https://github.com/LijunRio/VLM-Seminar25-Dataset?tab=readme-ov-file#-tasks--evaluation), but adapted for the model and to ensure **consistent output formats**.

## Setup:
1. Install scikit-learn, openai, and other needed packages. 
2. Clone the [VLM-Seminar25-Dataset](https://github.com/LijunRio/VLM-Seminar25-Dataset) repo into the `/code` folder.
3. Set your `NEBIUS_API_KEY` by creating a file at `config/user.env` and adding your API key. You can obtain a key from [Nebius Studio](https://studio.nebius.com/). Nebius provides $1 of free credit, which should be sufficient for all five notebooks. We are using the *Qwen2.5-VL-72B-Instruct* model.

