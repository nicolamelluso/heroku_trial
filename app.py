# coding: utf-8
"""
Example of a Streamlit app for an interactive spaCy model visualizer. You can
either download the script, or point streamlit run to the raw URL of this
file. For more details, see https://streamlit.io.
Installation:
pip install streamlit
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
python -m spacy download de_core_news_sm
Usage:
streamlit run streamlit_spacy.py
"""
from __future__ import unicode_literals

import streamlit as st
import spacy
from spacy import displacy
import pandas as pd

ner_displacy_palette = {
    "DEVICE": "#da3650",
    "USER": "#67328b",
    "TOOL": "#00a594",
    "BODY": "#fcd548",
    "GRIPPER": "#007bac",
    "OPERATION": "#6c63a5",
    "CONCEPT": "#df5a35",
}

ents = [
        "DEVICE",
        "TOOL",
        "USER",
        "BODY",
        "GRIPPER",
        "OPERATION",
        "CONCEPT",
    ]

ner_displacy_options = {
    "ents": ents,
    "colors": ner_displacy_palette,
}

SPACY_MODEL_NAMES = ["ergonomy_spacy_model"]
DEFAULT_TEXT = "This paper is the first review of haptic optical tweezers, a new technique which associates force feedback teleoperation with optical tweezers. \n\n This technique allows users to explore the microworld by sensing and exerting picoNewton-scale forces with trapped microspheres. Haptic optical tweezers also allow improved dexterity of micromanipulation and micro-assembly. One of the challenges of this technique is to sense and magnify picoNewton-scale forces by a factor of 1012 to enable human operators to perceive interactions that they have never experienced before, such as adhesion phenomena, extremely low inertia, and high frequency dynamics of extremely small objects. The design of optical tweezers for high quality haptic feedback is challenging, given the requirements for very high sensitivity and dynamic stability. The concept, design process, and specification of optical tweezers reviewed here are focused on those intended for haptic teleoperation. In this paper, two new specific designs as well as the current state-of-the-art are presented. Moreover, the remaining important issues are identified for further developments. The initial results obtained are promising and demonstrate that optical tweezers have a significant potential for haptic exploration of the microworld. Haptic optical tweezers will become an invaluable tool for force feedback micromanipulation of biological samples and nano- and micro-assembly parts."
import pandas as pd
docs = pd.read_excel('data/docs.xlsx')
#PAPER = docs[docs['docId']=='D003']['content'].values[0][:600]

#sents = pd.read_excel('sentences.xlsx')


SECOND_TEXT = "Micro- and nano-technologies are, in theory, very attractive. Theoretical models predict incredible properties for nano- and micro-structures. But in practice however, researchers and inventors face an unknown puzzle: there is no analogy in the macroworld that can prepare operators for the unpredictability and delicacy of the microscopic world. Barack Obama is the president of United States. Existing knowledge and know-how are experimentally insufficient. Exploration is the most delicate and unavoidable task for micromanipulation, microfabrication, and microassembly processes. In biology, the manipulation and the exploration of single cells or protein properties is a critical challenge.1,2 This can only be performed by an experienced user. These procedures are highly time-consuming and uneconomical."

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 0.5rem; margin-bottom: 0.5rem">{}</div>"""


@st.cache(ignore_hash=True)
def load_model(name):
    return spacy.load(name)


@st.cache(ignore_hash=True)
def process_text(model_name, text):
    nlp = load_model(model_name)
    return nlp(text)


st.sidebar.title("Ergonomics Named Entity Visualizer")
st.sidebar.markdown(
    """
Process text with [spaCy](https://spacy.io) models and visualize named entities relative to the _Ergonomy_ field. Uses spaCy's built-in
[displaCy](http://spacy.io/usage/visualizers) visualizer under the hood.
"""
)

spacy_model = st.sidebar.selectbox("Model name", SPACY_MODEL_NAMES)
model_load_state = st.info(f"Loading model '{spacy_model}'...")
nlp = load_model(spacy_model)
model_load_state.empty()

text = st.text_area("Text to analyze", DEFAULT_TEXT)
doc = process_text(spacy_model, text)


st.header("Entities")
st.sidebar.header("Named Entities")
#label_set = nlp.get_pipe("ner").labels
label_set = ['DEVICE','USER','TOOL','GRIPPER','BODY','OPERATION']
labels = st.sidebar.multiselect("Entity labels", label_set, label_set)
html = displacy.render(doc, style="ent", options={"ents": labels,"colors": ner_displacy_palette})
# Newlines seem to mess with the rendering
#html = html.replace("\n", " ")
st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)

paper = st.sidebar.selectbox("Paper", docs.docId.values)

doc_main = nlp(docs[docs['docId']==paper]['content'].values[0])

attrs = ["text", "label_", "start", "end", "start_char", "end_char"]
dfs = []

for sent in doc_main.sents:
    doc = process_text(spacy_model, sent.text)
    
    if any(elem in ents for elem in [ent.label_ for ent in doc.ents]) & ('\n' not in doc.text):
        html = displacy.render(doc, style="ent", options={"ents": labels,"colors": ner_displacy_palette})
        st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
        
        data = [
                [str(getattr(ent, attr)) for attr in attrs]
                for ent in doc.ents
                if ent.label_ in labels
                ]
        
        dfs.append(pd.DataFrame(data, columns=attrs))

df = pd.concat(dfs)

df = df.groupby(['text','label_']).count()[['start']].sort_values(by='start',ascending = False)
df = df.reset_index()
df.columns = ['Entity','Entity type','Frequency']

st.table(df)




if "textcat" in nlp.pipe_names:
    st.header("Text Classification")
    st.markdown(f"> {text}")
    df = pd.DataFrame(doc.cats.items(), columns=("Label", "Score"))
    st.dataframe(df)

