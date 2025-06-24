# 🌾 Agricultural Chatbot using Generative QA
![Gradio Chatbot UI](https://drive.google.com/uc?export=view&id=1SMfbClo5ihUgZIFW0YQ2JPgBzEfSfBqM)
![-](https://drive.google.com/uc?export=view&id=1f04Yx7NHeIhAjyhmA_UnUiDy6C0f_uw4)



This repository contains the implementation of an AI-powered Agricultural Chatbot designed to assist farmers with queries on crop diseases, pest control, soil fertility, and sustainable farming practices. Built using the T5-small Transformer model and fine-tuned on a domain-specific dataset, the chatbot provides contextual answers through a seamless Gradio-based interface.

---

##  Project Overview

- **Purpose**: Bridge the information gap for farmers, especially in rural areas, by offering timely and accessible agricultural knowledge.
- **Model**: T5-small fine-tuned on domain-specific agricultural QA data.
- **Dataset**: [KisanVaani/agriculture-qa-english-only](https://huggingface.co/datasets/KisanVaani/agriculture-qa-english-only) — 22,615 QA pairs (reduced to ~2,331 unique after deduplication).
- **Interface**: [Gradio](https://gradio.app/) web UI for natural, interactive chatbot conversations.

---

### Dataset

1. Source: Hugging Face Datasets Hub

2. Size: 22,615 QA pairs (reduced to ~2,331 unique pairs after deduplication)

3. Structure: question and answers
4. Content Coverage: Crop diseases, pest control, soil health, organic farming, weather considerations.

5. Limitations:

- Reduced diversity after duplicate removal


  ### Hyperparameter Tuning Results

| Experiment | Learning Rate | Batch Size | Epochs | Val Loss | Train Loss | Val Accuracy | Train Accuracy | Improvement Over Baseline |
|------------|---------------|------------|--------|----------|------------|---------------|----------------|----------------------------|
| 0 (Baseline) | 0.00003      | 8          | 5      | 1.5693   | 1.7723     | 0.7437        | 0.7189         | 0.00%                      |
| 1          | 0.00003        | 16         | 5      | 1.6833   | 2.0990     | 0.7358        | 0.6811         | -7.26%                     |
| 2          | 0.00003        | 8          | 20     | 1.2800   | 1.3802     | 0.7648        | 0.7526         | 18.43%                     |
| 3 (Best)   | 0.00002        | 4          | 30     | 1.2175   | 1.2782     | 0.7733        | 0.7650         | 22.42%                     |


The best-performing configuration (learning rate = 2e-5, batch size = 4, epochs = 30) achieved a validation accuracy of 77.33% with a 22.42% improvement over the baseline. Lower learning rate and smaller batch size allowed for more stable convergence, and extended training (30 epochs) helped the model generalise better.


  ### Performance Metrics (best-performing model)

| Metric       | Value     | Interpretation                            |
|--------------|-----------|--------------------------------------------|
| ROUGE-L      | 0.0400    | Low structural overlap with reference text |
| BLEU         |  0.0104    | Limited n-gram overlap in generation       |
| Val Accuracy | 0.765015     | From best configuration                    |
| Val Loss     | 22.4695%  | Improved from baseline after tuning        |

While the model achieved notable gains in validation accuracy, the low ROUGE-L and BLEU scores suggest that responses are fluent but often generic. This both maybe been caused by the limited training data (~2,300 unique pairs) and the modest capacity of T5-small. Future improvements may benefit from larger models and richer data.

### Example Conversation
- User: "How to control pests on maize?"
- Chatbot: "Use pesticides."

- Response is correct, but may lack specific detail. (Future goal: make answers more targeted)

---

###  Deployment

Deployed via Hugging Face Spaces.

 [Live Chatbot Demo](https://docs.google.com/document/d/1zxgSkirN0TaLzGcgyL4LFNXyzUNIetvXYgNzzPS9qlU/edit?usp=sharing)

 
###  Video

A Walkthrough Video presentation.

 [Video](https://docs.google.com/document/d/1zxgSkirN0TaLzGcgyL4LFNXyzUNIetvXYgNzzPS9qlU/edit?usp=sharing)


---

##  Setup Instructions

### 🔧 Prerequisites
- Python 3.8+
- pip
- TensorFlow >= 2.11
- Internet connection

### Clone & Install
```bash
git clone https://github.com/Joh-Ishimwe/Agri_chatbot.git
cd Agri_chatbot
pip install -r requirements.txt

```


1. Clone the Repository:
   
`git clone https://github.com/your-username/agri-chatbot
cd agri-chatbot
`

3. Install Dependencies:

 `pip install -r requirements.txt`

 
4. Download the Dataset: The dataset is automatically downloaded in the notebook using:

   `df = pd.read_parquet("hf://datasets/KisanVaani/agriculture-qa-english-only/data/train-00000-of-00001.parquet")`


5. Run the Jupyter Notebook cells:
 - Preprocess

- Train model

- Launch the Gradio Interface
   

### User Interface

- The chatbot interface, built with Gradio, offers:

- Simple layout with instructions

- Example questions for easy start

- Chat history tracking

- Input box + Send & Clear buttons



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

  
## Repository Structure

├── models/                                                                            # Saved trained models
├── notebook/                                                                          # Jupyter training & evaluation notebook
├── app.py                                                                             # Gradio app for chatbot
├── requirements.txt                                                                   # Python dependencies
├── README.md                                                                          # This file
├── Report-Agricultural                                                                # Final PDF report


## References

1. Scientific Article on Deep Learning in Agriculture

2. Adam vs AdamW Optimizer Discussion


