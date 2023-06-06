import streamlit as st

from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

st.cache(show_spinner=False)
def load_model():
    model_name = 'google/pegasus-large'
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name, max_position_embeddings=2000).to(torch_device)
    return model,tokenizer

model,tokenizer = load_model()

st.header("Prototyping an NLP solution")
st.text("This demo uses a model for Question Answering.")
add_text_sidebar = st.sidebar.title("Menu")
add_text_sidebar = st.sidebar.text("Just some random text.")
src_text = st.text_input(label='Insert a question.')

if(st.button("Generate")):
    batch = tokenizer.prepare_seq2seq_batch(src_text, truncation=True, padding='longest',return_tensors='pt')
    translated = model.generate(**batch,min_length=50)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    st.text_area('Summary: ', tgt_text)