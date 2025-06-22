import gradio as gr
from transformers import T5Tokenizer, TFT5ForConditionalGeneration
import tensorflow as tf
import re
import os

# Setup function to load trained model with fallback
def setup_model():
    trained_model_path = "./models/lr_2e-05_bs_4_ep_30"  # Path relative to Space
    try:
        if os.path.exists(trained_model_path):
            print("Loading trained model from:", trained_model_path)
            model = TFT5ForConditionalGeneration.from_pretrained(trained_model_path)
        else:
            print("Trained model path not found. Using base T5-small model.")
            model = TFT5ForConditionalGeneration.from_pretrained("t5-small")
    except Exception as e:
        print(f"Error loading trained model: {e}. Falling back to base T5-small model.")
        model = TFT5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    return model, tokenizer

# Load model and tokenizer globally
model, tokenizer = setup_model()

def predict(question):
    """Generate a response using the trained model with aligned prompting and adjusted parameters."""
    input_text = f"question: {question}"
    inputs = tokenizer(input_text, return_tensors="np", max_length=128, truncation=True, padding=True)['input_ids']
    outputs = model.generate(
        input_ids=tf.convert_to_tensor(inputs),
        max_length=200,
        min_length=30,
        num_beams=8,
        no_repeat_ngram_size=3,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        early_stopping=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if response.lower().startswith(question.lower().split()[0]):
        response = "A possible answer: " + response
    return response

# Example questions
example_questions = [
    "What is crop rotation and why is it important?",
    "How can I prevent soil erosion on my farm?",
    "What are the signs of nitrogen deficiency in plants?",
    "How do I control aphids naturally?",
    "When is the best time to plant tomatoes?",
    "What causes yellowing leaves in crops?",
    "How can I improve soil fertility organically?",
    "What are companion plants for corn?"
]

# Custom CSS
custom_css = """
.gradio-container {
    max-width: 800px !important;
    margin: auto !important;
}
.chat-message {
    font-size: 16px !important;
}
.message-wrap {
    max-width: 80% !important;
}
"""

# Gradio interface
with gr.Blocks(
    css=custom_css,
    title="🌱 Agro-Bot Assistant",
    theme=gr.themes.Soft(
        primary_hue="green",
        secondary_hue="emerald",
        neutral_hue="slate",
    )
) as iface:
    gr.HTML("""
        <div style="color: #2d5a27; text-align: center; padding: 20px;">
            <h1 style="margin-bottom: 10px;">
                🌱 Agro-Bot: Your Agricultural Assistant
            </h1>
            <p style="font-size: 18px; color: #666; margin-bottom: 20px;">
                Get expert advice on farming, crop management, pest control, and soil health
            </p>
        </div>
    """)
    with gr.Row():
        gr.HTML("""
            <div style="background: linear-gradient(135deg, #e8f5e8 0%, #f0f9f0 100%);
                        padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <h3 style="color: #2d5a27; margin-top: 0;">📋 How to Use Agro-Bot:</h3>
                <ul style="color: #444; line-height: 1.6;">
                    <li>💬 Type your agricultural question in the chat box below</li>
                    <li>🎯 Be specific for better answers (mention crops, conditions, etc.)</li>
                    <li>📚 Ask about crop management, pest control, soil health, or farming techniques</li>
                    <li>🔄 Continue the conversation for follow-up questions</li>
                </ul>
            </div>
        """)
    chatbot = gr.Chatbot(
        height=400,
        placeholder="👋 Hello! I'm Agro-Bot, ready to help with your agricultural questions!",
        show_label=False,
        container=True
    )
    with gr.Row():
        msg = gr.Textbox(
            placeholder="💭 Ask me anything about agriculture... (e.g., 'How do I prevent tomato blight?')",
            show_label=False,
            scale=4,
            container=False
        )
        submit_btn = gr.Button("Send 🚀", scale=1, variant="primary")
    gr.HTML("<h3 style='color: #2d5a27; margin-top: 30px;'>💡 Try these example questions:</h3>")
    with gr.Row():
        example_col1 = gr.Column(scale=1)
        example_col2 = gr.Column(scale=1)
    with example_col1:
        for i in range(0, len(example_questions), 2):
            gr.Button(example_questions[i], size="sm", variant="secondary").click(
                lambda x=example_questions[i]: x, outputs=msg
            )
    with example_col2:
        for i in range(1, len(example_questions), 2):
            if i < len(example_questions):
                gr.Button(example_questions[i], size="sm", variant="secondary").click(
                    lambda x=example_questions[i]: x, outputs=msg
                )
    gr.HTML("""
        <div style="text-align: center; margin-top: 30px; padding: 15px;
                    background-color: #f8f9fa; border-radius: 8px;">
            <p style="color: #666; margin: 0;">
                🌾 <strong>Disclaimer:</strong> This AI assistant provides general agricultural guidance.
                Always consult with local agricultural experts for specific farming decisions.
            </p>
        </div>
    """)
    def respond(message, chat_history):
        if message.strip() == "":
            return gr.update(value=""), chat_history
        bot_message = predict(message)
        if not isinstance(chat_history, list):
            chat_history = []
        chat_history.append([message, bot_message])
        return gr.update(value=""), chat_history
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    submit_btn.click(respond, [msg, chatbot], [msg, chatbot])
    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])

# Launch (handled by Spaces, not included in app.py)
if __name__ == "__main__":
    iface.launch()