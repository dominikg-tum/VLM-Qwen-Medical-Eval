# VLM-Seminar-Final-Submission
Master Seminar Submission for AI in Vision-Language Models in Medical Imaging.

## Structure:
For the two datasets (1) chest-xrays and (2) brain-mris, there are are two / three respective notebooks to solve the tasks of (1) classification and grounding and (2) detection, diagnosis, and description using the VLM *Qwen2.5-VL-72B-Instruct*. 

Results of the predicitons and evaluations are saved under `./results/"chest_xrays or nova_brain"/*.json`.

## Setup:
1. Install scikit-learn, openai, and other needed packages. 
2. Clone the [VLM-Seminar25-Dataset](https://github.com/LijunRio/VLM-Seminar25-Dataset) repo into the `/code` folder.
3. Set your `NEBIUS_API_KEY` by creating a file at `config/user.env` and adding your API key. You can obtain a key from [Nebius Studio](https://studio.nebius.com/). Nebius provides $1 of free credit, which should be sufficient for all five notebooks. We are using the *72B VL* model.

