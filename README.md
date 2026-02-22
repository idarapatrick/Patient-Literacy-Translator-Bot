# The Patient Literacy Translator: Fine-Tuning Gemma-2B for Medical Jargon Simplification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/idarapatrick/Patient-Literacy-Translator-Bot/blob/main/Medical_Translator.ipynb)

## Project Definition and Domain Alignment
This project features a domain-specific Large Language Model customized for the healthcare sector. Medical professionals consistently document patient encounters using dense clinical terminology. When patients read their discharge summaries or lab results, this vocabulary creates a significant barrier to health literacy. Misunderstanding medical instructions can lead to poor health outcomes and improper medication usage. 

To solve this problem, this repository contains a complete end-to-end pipeline for building a Patient Literacy Translator. The assistant takes complex medical text and translates it into plain English suitable for a general audience. The project provides a single Jupyter Notebook designed to run seamlessly on Google Colab, covering data preprocessing, parameter-efficient fine-tuning, evaluation, and user interface deployment.

## Dataset Collection and Preprocessing
Training an accurate translation model requires high-quality, domain-specific instruction data. This project utilizes the Medical Meadow Wikidoc Patient Information dataset sourced from Hugging Face. The data consists of complex medical descriptions paired with simplified patient-friendly explanations. 

To prepare the data for training, a subset of 2,500 examples was extracted to balance training efficiency with model performance. The preprocessing pipeline removed missing values and applied regular expressions to strip out stray HTML tags that would otherwise introduce noise. Because the chosen model requires specific control tokens to understand conversational turns, the data was formatted into strict instruction and response templates using `<start_of_turn>user` and `<start_of_turn>model` tags. Subword tokenization was then applied to break down rare medical terminology into smaller, frequent subwords that the language model can easily process.

## Model Architecture and Fine-Tuning Methodology
The base model selected for this project is `google/gemma-2-2b-it`. This model offers strong generative capabilities but requires significant computational power for a full-weight update. To ensure the training pipeline could execute efficiently on Google Colab's free T4 GPU resources, Parameter-Efficient Fine-Tuning was implemented using the Hugging Face PEFT library.

Specifically, Low-Rank Adaptation (LoRA) was applied to the base model. This technique freezes the original pre-trained weights and injects trainable rank decomposition matrices into the attention layers, targeting the query, value, and projection modules with a rank of 8 and an alpha of 16. This approach reduced the number of trainable parameters by over 99 percent. Additionally, 4-bit quantization was integrated via the BitsAndBytes library to compress the model's memory footprint and prevent out-of-memory errors during training.

## Hyperparameter Optimization
Finding the optimal training configuration required structured experimentation. Three distinct training runs were conducted to evaluate the impact of different hyperparameters on the model's convergence. 

The baseline experiment utilized a learning rate of 2e-4 and a batch size of 2. A second experiment lowered the learning rate to 5e-5 to test for better generalization. The final experiment maintained the 2e-4 learning rate but increased the effective batch size to 4 using gradient accumulation steps. This final configuration yielded the most stable training loss and was selected to train the production model for one complete epoch. Training for a single epoch ensured the model learned the translation format without catastrophic forgetting of its foundational English capabilities.

## Performance Metrics
The fine-tuned model was evaluated quantitatively using the ROUGE metric against a holdout set of human-written references. The evaluation script generated a ROUGE-1 score of 0.2436, a ROUGE-2 score of 0.0731, and a ROUGE-L score of 0.1801. 

While ROUGE-1 confirms the model successfully captures core vocabulary, the ROUGE-L score is the most critical metric for this domain. It evaluates the longest common subsequence, confirming that the overall sentence structure of the AI-generated text aligns closely with natural human phrasing rather than simply swapping medical words for simple words.

## Impact of Fine-Tuning and Inference Demonstration
Prior to fine-tuning, the base Gemma model struggled with the translation task. When given a medical prompt, the base model would frequently generate lengthy encyclopedia articles or simply repeat the clinical facts without lowering the reading level. The fine-tuned model actively restructures the sentences to abstract the medical jargon completely.

**Original Complex Input:**
> MRI of the lumbar spine indicates severe spinal stenosis at the L4-L5 level with bilateral foraminal narrowing. This is causing radicular pain radiating down the posterior aspect of the lower extremities. Surgical intervention via laminectomy may be indicated if conservative management fails.

**Fine-Tuned Simplified Output:**
> MRI of the lumbar spine shows severe spinal stenosis at the L4-L5 level with bilateral foraminal narrowing. This is causing pain that radiates down the back of the lower leg. If conservative treatment does not work, surgery may be needed.

## User Interface Integration
To make the fine-tuned model accessible, it is deployed within an interactive web application using Gradio. The interface abstracts the underlying inference code, providing users with a simple text box to paste their clinical notes and immediately receive a translated output.

## Steps to Run the Model
The repository is designed for immediate execution with minimal environment configuration.

1. Click the "Open in Colab" badge at the top of this repository to launch the environment.
2. Navigate to the Colab Secrets manager (the key icon on the left sidebar) and add your Hugging Face access token as a new secret named `HF_TOKEN`.
3. Execute the notebook cells in sequential order. The script will automatically download the dataset, execute the preprocessing functions, apply the LoRA adapters, and train the model.
4. The final cell in the notebook will evaluate the model and launch a public Gradio web link where you can interact with the Patient Literacy Translator directly.