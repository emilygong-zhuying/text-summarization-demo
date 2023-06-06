import streamlit as st

from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

st.cache(show_spinner=False)
def load_model():
    model_name = 'google/pegasus-large'
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name, max_position_embeddings=2000).to(torch_device)
    #run using local model
    #tokenizer = PegasusTokenizer.from_pretrained(local_pegasus-large_tokenizer)
    #model = PegasusForConditionalGeneration.from_pretrained(local_pegasus-large_tokenizer_model, max_position_embeddings=2000).to(torch_device)
    return model,tokenizer

#run this the first time and use the local model for faster runtime
#tokenizer.save_pretrained("local_pegasus-large_tokenizer")
#model.save_pretrained("local_pegasus-large_tokenizer_model")
model,tokenizer = load_model()

st.header("Abstractive Summurization with PEGASUS-LARGE")
st.text("Try inputting a prompt below!")
src_text = st.text_input(placeholder='In mathematics, a metric space is a set together with a notion of distance between its elements, usually called points. The distance is measured by a function called a metric or distance function. Metric spaces are the most general setting for studying many of the concepts of mathematical analysis and geometry.The most familiar example of a metric space is 3-dimensional Euclidean space with its usual notion of distance. Other well-known examples are a sphere equipped with the angular distance and the hyperbolic plane. A metric may correspond to a metaphorical, rather than physical, notion of distance: for example, the set of 100-character Unicode strings can be equipped with the Hamming distance, which measures the number of characters that need to be changed to get from one string to another.Since they are very general, metric spaces are a tool used in many different branches of mathematics. Many types of mathematical objects have a natural notion of distance and therefore admit the structure of a metric space, including Riemannian manifolds, normed vector spaces, and graphs. In abstract algebra, the p-adic numbers arise as elements of the completion of a metric structure on the rational numbers. Metric spaces are also studied in their own right in metric geometry and analysis on metric spaces.Many of the basic notions of mathematical analysis, including balls, completeness, as well as uniform, Lipschitz, and Hölder continuity, can be defined in the setting of metric spaces. Other notions, such as continuity, compactness, and open and closed sets, can be defined for metric spaces, but also in the even more general setting of topological spaces.', label="Input Text")

if(st.button("Generate")):
    batch = tokenizer.prepare_seq2seq_batch(src_text, truncation=True, padding='longest',return_tensors='pt')
    translated = model.generate(**batch,min_length=50)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    st.write("#### Summary:")
    st.write(tgt_text[0], unsafe_allow_html=True)
else:
    pass