# verkiezingsdebat_full_pipeline.py

# Volledig prototype: van audio/video tot analyse en dashboard

# Versie aangepast voor Streamlit Cloud zonder en_core_web_sm

import os
import pandas as pd
from textblob import TextBlob
from transformers import pipeline
import streamlit as st
import whisper
import spacy

# Gebruik een lege Engelse SpaCy pipeline (werkt op Streamlit Cloud)

nlp = spacy.blank("en")

# NLP modellen voor emoties

emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# -----------------------------

# Functies voor transcriptie

# -----------------------------

def transcribe_audio(file_path):
model = whisper.load_model("base")
result = model.transcribe(file_path)
return result["text"]

# -----------------------------

# Functies voor analyse

# -----------------------------

def analyze_emotion(text):
return emotion_analyzer(text)

def analyze_sentiment(text):
blob = TextBlob(text)
return {"polarity": blob.sentiment.polarity, "subjectivity": blob.sentiment.subjectivity}

def detect_framing(text):
framing_keywords = ["crisis", "gevaar", "dreiging", "falen", "wonder", "zekerheid", "belofte"]
doc = nlp(text)
return [token.text for token in doc if token.text.lower() in framing_keywords]

def detect_facts(text):
doc = nlp(text)
# gebruik zinsplitsing met sents (werkt met lege pipeline)
fact_sentences = [sent.text for sent in doc.sents if any(char.isdigit() for char in sent.text) or "volgens" in sent.text.lower()]
return fact_sentences

def check_facts(fact_sentences):
results = []
for sent in fact_sentences:
results.append({"sentence": sent, "fact_checked": "nog niet geverifieerd"})
return results

def analyze_debate(text):
emotion = analyze_emotion(text)
sentiment = analyze_sentiment(text)
framing = detect_framing(text)
facts = detect_facts(text)
fact_check = check_facts(facts)

```
return {
    "emotions": emotion,
    "sentiment": sentiment,
    "framing_keywords": framing,
    "fact_sentences": facts,
    "fact_check": fact_check
}
```

# -----------------------------

# Streamlit dashboard

# -----------------------------

st.title("Volledig Verkiezingsdebat Analyse Dashboard")
st.markdown("""
Upload audio/video of tekstbestanden van verkiezingsdebatten.
Het systeem analyseert op:

* Emoties
* Sentiment
* Framing
* Feitelijkheid
  """)

uploaded_files = st.file_uploader("Upload debatten (.txt, .mp3, .wav, .mp4)", accept_multiple_files=True)

if uploaded_files:
all_results = []
for file in uploaded_files:
file_ext = os.path.splitext(file.name)[1].lower()
if file_ext in [".txt"]:
text = file.read().decode("utf-8")
else:
temp_path = f"temp_{file.name}"
with open(temp_path, "wb") as f:
f.write(file.read())
text = transcribe_audio(temp_path)
os.remove(temp_path)

```
    result = analyze_debate(text)
    all_results.append({"filename": file.name, **result})

for res in all_results:
    st.header(f"Debat: {res['filename']}")
    st.subheader("Sentiment")
    st.write(res['sentiment'])
    st.subheader("Emoties")
    st.write(res['emotions'])
    st.subheader("Framing Keywords")
    st.write(res['framing_keywords'])
    st.subheader("Feiten (ongeverifieerd)")
    st.write(res['fact_sentences'])
    st.subheader("Fact-check Status")
    st.write(res['fact_check'])

df = pd.DataFrame([{
    "filename": r["filename"],
    "polarity": r["sentiment"]["polarity"],
    "subjectivity": r["sentiment"]["subjectivity"],
    "framing_count": len(r["framing_keywords"]),
    "fact_count": len(r["fact_sentences"])
} for r in all_results])

st.subheader("Overzicht per debat")
st.bar_chart(df.set_index("filename")[["polarity", "subjectivity", "framing_count", "fact_count"]])
```
