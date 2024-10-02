import streamlit as st
from PyPDF2 import PdfReader
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from googletrans import Translator

# Load environment variables
load_dotenv()

# Configure Google GenAI API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

def get_pdf_text(pdf_docs):
    texts = []
    for pdf in pdf_docs:
        pdf_bytes = pdf.read()  # Read the uploaded PDF file contents
        pdf_reader = PdfReader(BytesIO(pdf_bytes))  # Create a PdfReader from the bytes
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        texts.append(text)
    return texts

def get_text_chunks(texts):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = []
    for text in texts:
        chunks.extend(text_splitter.split_text(text))
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store.save_local("faiss_index")

def get_summarization_chain():
    prompt_template = PromptTemplate(template="Summarize the text in 100-150 words: {context}", input_variables=["context"])
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    qa_chain = load_qa_chain(model, chain_type="stuff", prompt=prompt_template)
    return qa_chain

def summarize_pdf(pdf_docs):
    raw_texts = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_texts)
    get_vector_store(text_chunks)

    chain = get_summarization_chain()

    # Create a list of documents with page_content and metadata attributes
    documents = [Document(page_content=chunk, metadata={}) for chunk in text_chunks]

    inputs = {
        "input_documents": documents,
        "task": "summarization"
    }

    summary = chain(inputs, return_only_outputs=True)
    return summary["output_text"]

def extract_transcript_details(youtube_video_url):
    try:
        video_id = youtube_video_url.split("=")[1]
        languages = ['en', 'es', 'fr', 'de', 'it', 'kn', 'hi', 'te', 'ml', 'ta']  # Add more languages as needed
        for language in languages:
            try:
                transcript_text = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
                transcript_text = " ".join([i["text"] for i in transcript_text])
                return transcript_text
            except Exception as e:
                if "No transcripts were found for any of the requested language codes" in str(e):
                    continue
                else:
                    st.error(f"Error extracting transcript: {e}")
                    return None
        st.error("No transcript found for this video in any of the supported languages.")
        return None
    except ValueError as e:
        st.error(f"Invalid YouTube video URL: {e}")
        return None
    except Exception as e:
        st.error(f"Error extracting transcript: {e}")
        return None

def generate_gemini_content(transcript_text, prompt):
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt + transcript_text)
        
        # Check if the response contains a valid Part
        if response.parts:
            return response.text
        else:
            st.error("Invalid response from GenAI API. Please check the candidate.safety_ratings for more information.")
            return None
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return None

def translate_text(text, target_language):
    try:
        translator = Translator()
        translation = translator.translate(text, dest=target_language)
        return translation.text
    except Exception as e:
        st.error (f"Error translating text: {e}")
        return None

def main():
    st.set_page_config("PDF and YouTube Video Summarizer")
    st.header("QuickCap (PDF and youtube summarizer)by Tech Terrific")

    col1 , col2 = st.columns(2)

    with col1:
        pdf_docs = st.file_uploader("Upload a PDF", type="pdf", accept_multiple_files=True)
        if pdf_docs:  # Check if files were uploaded
            if st.button("Summarize PDF"):
                with st.spinner("Processing..."):
                    summary = summarize_pdf(pdf_docs)
                    st.success("Done")
                    st.write("Summary:", summary)

    with col2:
        youtube_link = st.text_input("Enter Any language YouTube video Link:")
        if youtube_link:
            try:
                video_id = youtube_link.split("=")[1]
                print(video_id)
                st.image(f"https://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)
                if st.button("Summarize YouTube Video"):
                    with st.spinner("Processing..."):
                        transcript_text = extract_transcript_details(youtube_video_url=youtube_link)
                        if transcript_text:
                            prompt = """You're youtube video summarizer. You will be taking the transcript text and summarising the either video or providing the important summary in Points Within 10,000 words. Please provide the summary of the text given here"""
                            summary = generate_gemini_content(transcript_text, prompt)
                            if summary:
                                st.write(summary)
                            else:
                                st.error("Failed to generate summary")
                        else:
                            st.error("Failed to extract transcript")
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()