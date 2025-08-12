import streamlit as st
import pandas as pd
import spacy

# ✅ Load spaCy model at runtime if not present
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

import neattext as nt
import neattext.functions as nfx
from collections import Counter
import base64
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
from textblob import TextBlob
from wordcloud import WordCloud
import docx2txt
from PyPDF2 import PdfReader


# ------------------- Utility Functions -------------------
def text_analysis(my_text):
    docx = nlp(my_text)
    alldata = [(token.text, token.lemma_, token.pos_, token.tag_,
                token.shape_, token.is_alpha, token.is_stop) for token in docx]
    df = pd.DataFrame(alldata,
                      columns=['Token', 'Lemma', 'PoS', 'Tag', 'Shape', 'Is_Alpha', 'Is_Stopword'])
    return df

def get_most_common_tokens(my_text, num=4):
    word_tokens = Counter(my_text.split())
    return dict(word_tokens.most_common(num))

def get_sentiment(my_text):
    blob = TextBlob(my_text)
    return blob.sentiment

def plot_get_wordcloud(my_text):
    from matplotlib import pyplot as plt
    my_wordcloud = WordCloud().generate(my_text)
    fig = plt.figure()
    plt.imshow(my_wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(fig)

def plot_word_freq(keywords):
    from matplotlib import pyplot as plt
    fig = plt.figure()
    plt.bar(keywords.keys(), keywords.values())
    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_pos_counts(df):
    import seaborn as sns
    from matplotlib import pyplot as plt
    fig = plt.figure()
    sns.countplot(df['PoS'])
    plt.xticks(rotation=45)
    st.pyplot(fig)

def make_downloaddable(data):
    csvfile = data.to_csv(index=False)
    b64 = base64.b64encode(csvfile.encode()).decode()
    new_filename = f"nlp_result_{timestr}.csv"
    st.markdown("###⬇️📥 Download CSV file###")
    href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Click here to download</a>'
    st.markdown(href, unsafe_allow_html=True)

def read_pdf(file):
    pdfReader = PdfReader(file)
    count = len(pdfReader.pages)
    all_page_text = ""
    for i in range(count):
        page = pdfReader.pages[i]
        all_page_text += page.extract_text()
    return all_page_text

def entities(my_text):
    docx = nlp(my_text)
    return [(ent.text, ent.label_) for ent in docx.ents]


# ------------------- Streamlit Application -------------------
def application():
    st.title("NLP Application")
    menu = ["Home", "NLP(files)", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home - Analyse text")
        raw_text = st.text_area("Enter text here")
        num_of_most_common = st.sidebar.number_input("Most common words", 5, 15)

        if st.button("Analyse"):
            with st.expander("Original text"):
                st.write(raw_text)

            with st.expander("Text Analysis"):
                token_result_df = text_analysis(raw_text)
                st.dataframe(token_result_df)

            with st.expander("Entities"):
                st.write(entities(raw_text))

            col1, col2 = st.columns(2)

            with col1:
                with st.expander("Word stats"):
                    st.info("Word Statistics")
                    docx = nt.TextFrame(raw_text)
                    st.write(docx.word_stats())

                with st.expander("Top Keywords"):
                    st.info("Top Keywords/Tokens")
                    processed_text = nfx.remove_stopwords(raw_text)
                    keywords = get_most_common_tokens(processed_text)
                    st.write(keywords)

                with st.expander("Sentiments"):
                    st.write(get_sentiment(raw_text))

            with col2:
                with st.expander("Plot Word Freq"):
                    processed_text = nfx.remove_stopwords(raw_text)
                    top_keywords = get_most_common_tokens(processed_text)
                    plot_word_freq(top_keywords)

                with st.expander("Plot Parts of Speech"):
                    plot_pos_counts(token_result_df)

                with st.expander("Plot WordCloud"):
                    plot_get_wordcloud(raw_text)

            with st.expander("Download Text Analysis Results"):
                make_downloaddable(token_result_df)

    elif choice == "NLP(files)":
        st.subheader("NLP task")
        text_file = st.file_uploader("Upload a text file", type=["txt", "pdf", "docx"])
        num_of_most_common = st.sidebar.number_input("Most common words", 5, 15)

        if text_file is not None:
            if text_file.type == "application/pdf":
                raw_text = read_pdf(text_file)
                st.write(raw_text)
            elif text_file.type == "text/plain":
                raw_text = str(text_file.read(), "utf-8")
                st.write(raw_text)
            else:
                raw_text = docx2txt.process(text_file)
                st.write(raw_text)

            if st.button("Analyse"):
                with st.expander("Original text"):
                    st.write(raw_text)

                with st.expander("Text Analysis"):
                    token_result_df = text_analysis(raw_text)
                    st.dataframe(token_result_df)

                with st.expander("Entities"):
                    st.write(entities(raw_text))

                col1, col2 = st.columns(2)

                with col1:
                    with st.expander("Word stats"):
                        st.info("Word Statistics")
                        docx = nt.TextFrame(raw_text)
                        st.write(docx.word_stats())

                    with st.expander("Top Keywords"):
                        st.info("Top Keywords/Tokens")
                        processed_text = nfx.remove_stopwords(raw_text)
                        keywords = get_most_common_tokens(processed_text , num_of_most_common)
                        st.write(keywords)

                    with st.expander("Sentiments"):
                        st.write(get_sentiment(raw_text))

                with col2:
                    with st.expander("Plot Word Freq"):
                        processed_text = nfx.remove_stopwords(raw_text)
                        top_keywords = get_most_common_tokens(processed_text , num_of_most_common)
                        plot_word_freq(top_keywords)

                    with st.expander("Plot Parts of Speech"):
                        plot_pos_counts(token_result_df)

                    with st.expander("Plot WordCloud"):
                        plot_get_wordcloud(raw_text)

                with st.expander("Download Text Analysis Results"):
                    make_downloaddable(token_result_df)

    else:
        st.subheader("About")


if __name__ == "__main__":
    application()
