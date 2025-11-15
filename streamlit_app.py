import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

@st.cache_resource
def load_model():
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

def generate_response(user_input, chat_history_ids=None):
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    
    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1)
    else:
        bot_input_ids = input_ids

    output_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        do_sample=True,
        top_p=0.92,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(output_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, output_ids

st.title("Conversational AI Chatbot (DialoGPT)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = None

if "messages" not in st.session_state:
    st.session_state.messages = []

user_message = st.text_input("You:")

if st.button("Send"):
    if user_message.strip():
        st.session_state.messages.append(("You", user_message))
        response, chat_history = generate_response(user_message, st.session_state.chat_history)
        st.session_state.chat_history = chat_history
        st.session_state.messages.append(("Bot", response))

for speaker, msg in st.session_state.messages:
    st.write(f"**{speaker}:** {msg}")
