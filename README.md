# Agricultural Chatbot using Generative QA

This repository contains the code and documentation for an AI-powered Agricultural Chatbot designed to assist farmers with queries on crop diseases, pest control, weather, and sustainable farming practices. Built using the T5-small Transformer model and fine-tuned on the KisanVaani dataset, the chatbot provides domain-specific answers through a user-friendly Gradio interface.

## Project Overview

**Purpose**: To provide accessible agricultural knowledge to farmers, especially in rural areas, enhancing decision-making and productivity.
**Model**: T5-small, a compact Transformer model fine-tuned for generative question-answering (QA).
**Dataset**: KisanVaani/agriculture-qa-english-only, containing 22,615 English question-answer pairs.
**Interface**: Gradio, offering an intuitive UI with chat history, example questions, and clear instructions.

## Dataset

- The chatbot is trained on the KisanVaani/agriculture-qa-english-only dataset from Hugging Face.





- Size: 22,615 question-answer pairs (reduced to ~2,331 unique pairs after deduplication).



- Structure: Two columns: question (e.g., "How to control pests on maize?") and answers (e.g., "Use organic pesticides like neem oil.").



- Content: Covers agricultural topics like crop diseases, pest management, soil fertility, and farming practices.



- Limitations: English-only, potentially limiting use in multilingual regions; reduced diversity due to high duplicate count.

- Performance Metrics

- The fine-tuned T5-small model was evaluated using standard NLP metrics and qualitative testing:





- ROUGE-L: 0.0507, indicating low structural similarity with reference answers due to generic responses.



- BLEU: 0.0804, showing limited n-gram overlap, reflecting challenges in capturing precise agricultural details.



- Validation Accuracy: 0.7736 (best configuration), with a 21.53% improvement in validation loss over the baseline.



- Qualitative Findings: Responses are fluent but often generic or lack specific details, as seen in example conversations below.

Note: Low metric scores suggest the model’s current limitations, attributed to T5-small’s capacity and the reduced dataset size after deduplication. Future improvements are outlined in the Future Work section.

## Deployed link and Video link: https://docs.google.com/document/d/1zxgSkirN0TaLzGcgyL4LFNXyzUNIetvXYgNzzPS9qlU/edit?usp=sharing


### Prerequisites





- Python 3.8+



- GPU (optional, recommended for faster training; Google Colab compatible)



- Internet access (to download dataset and pre-trained model)

- Setup Instructions

- Follow these steps to set up and run the chatbot locally or on Google Colab.





1. Clone the Repository:
   
`git clone https://github.com/your-username/agri-chatbot
cd agri-chatbot
`

3. Install Dependencies:

 `pip install -r requirements.txt`

 
4. Download the Dataset: The dataset is automatically downloaded in the notebook using:

   `df = pd.read_parquet("hf://datasets/KisanVaani/agriculture-qa-english-only/data/train-00000-of-00001.parquet")`

5. Run the Jupyter Notebook cells
6. Launch the Gradio Interface:
   - The final cells in the notebook start the Gradio UI. Access it via the provided URL (e.g., http://localhost:7860 locally or a public URL on Colab).



- Interact with the chatbot by typing questions or using example buttons.

## Example Conversations

Below is a sample interaction showcasing the chatbot’s functionality and current performance.

- Example 1: In-Domain Question
- User: "How to control pests on maize?"
- Chatbot: "Use pesticides."
- Analysis: The response is correct but generic.

## Challenges and Solutions





- Compatibility Issues: TensorFlow mismatches prevented using AdamW and label smoothing. Solution: Switched to tf.keras.optimizers.Adam for stable training.



- Limited Performance: Low ROUGE-L (0.0507) and BLEU (0.0804) scores due to T5-small’s capacity and dataset size. Solution: Improved validation loss by 21.53% via hyperparameter tuning.



- Resource Constraints: Google Colab’s memory limits restricted batch sizes. Solution: Used T5-small and batch sizes of 4-16 with early stopping.

Future Work

- To enhance the chatbot’s performance, consider:





- Extended Training: Train for 70-100 epochs with advanced learning rate schedules.



- Larger Models: Fine-tune T5-base or T5-large for better capacity.



- Dataset Expansion: Add diverse QA pairs or use augmentation (e.g., paraphrasing).



- Human Evaluation: Assess fluency and accuracy with farmer feedback.


