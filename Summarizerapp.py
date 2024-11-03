import re
import PyPDF2
from newspaper import Article
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import streamlit as st
import validators


def extract_text_from_url(link):
    article = Article(link)
    article.download()
    article.parse()
    return article.text

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text())
    return pages

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_pegasus_model(model_name):
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

def summarize_text(text, tokenizer, model, num_sentences):
    tokens = tokenizer(text, truncation=True, padding="longest", return_tensors="pt")
    summary_ids = model.generate(
        tokens['input_ids'],
        max_length=num_sentences * 4,
        min_length=num_sentences * 2,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True,
        do_sample=True,
        no_repeat_ngram_size=3,  # Prevents repetition of n-grams
        top_k=40,  # Use top-k sampling
        top_p=0.8  # Use nucleus sampling
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Add custom CSS for a border around the application
st.markdown(
    """
    <style>
    .main {
        border: 2px solid #4CAF50; /* Green border */
        padding: 10px;
        border-radius: 10px;
        margin: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Center the title using Markdown and HTML
st.markdown("<h1 style='text-align: center;'>Summarizer</h1>", unsafe_allow_html=True)

# Move Streamlit functions to the sidebar
st.sidebar.title("Input Options")
option = st.sidebar.radio(
    'How would you like to provide input?',
    ('News Article Link', 'Text Input', 'File Upload')
)

if option in ['News Article Link', 'Text Input']:
    if option == 'News Article Link':
        source_data = st.sidebar.text_input("Enter the news article link:")
    else:
        source_data = st.sidebar.text_area("Enter your text:")
    
    num_sentences = st.sidebar.slider("Select maximum number of words for summary:", 10, 100, 50)

elif option == 'File Upload':
    source_data = st.sidebar.file_uploader("Upload a file", type="pdf")
    summary_type = st.sidebar.radio("Choose summary type:", ('Page-wise Summary', 'Whole File Summary'))
    if summary_type == 'Whole File Summary':
        st.sidebar.warning("Note: Generating a whole file summary may take more time.")

    if source_data:
        pages = extract_text_from_pdf(source_data)
        
        if summary_type == 'Page-wise Summary':
            num_pages = st.sidebar.slider("Select number of pages to summarize:", 1, len(pages), len(pages))
            st.sidebar.info("Summarization in progress...")
            
            if num_pages > 0:
                if st.sidebar.button('SUMMARIZE', key='page_wise_summarize'):
                    with st.spinner('Preparing summary. Please wait...'):
                        tokenizer, model = load_pegasus_model("google/pegasus-large")
                        progress_text = st.empty()  # Create a placeholder for progress updates
                        
                        for i, page_text in enumerate(pages[:num_pages], 1):
                            preprocessed_text = preprocess_text(page_text)
                            summary = summarize_text(preprocessed_text, tokenizer, model, 10)
                            
                            # Update progress
                            progress_text.info(f"Generated summaries for {i} out of {num_pages} pages...")
                            
                            st.subheader(f"Summary of Page {i}:")
                            st.write(summary)
                            st.markdown("---")
                        
                        # Clear the progress message and show final success message
                        progress_text.empty()
                        st.success("All summaries generated successfully!")

        else:  # Whole File Summary
            if st.sidebar.button('SUMMARIZE', key='whole_file_summarize'):
                with st.spinner('Preparing summary. Please wait...'):
                    tokenizer, model = load_pegasus_model("google/pegasus-large")
                    full_text = ' '.join(pages)
                    preprocessed_text = preprocess_text(full_text)
                    full_summary = summarize_text(preprocessed_text, tokenizer, model, 20)
                    
                    st.success("Summary generated successfully!")
                    st.subheader("Whole File Summary:")
                    st.write(full_summary)

if st.sidebar.button('SUMMARIZE'):
    with st.spinner('Preparing summary. Please wait...'):
        try:
            summary_displayed = False  # Flag to check if a summary has been displayed
            
            if option == 'News Article Link':
                if not source_data or not validators.url(source_data):
                    st.error("Please enter a valid URL")
                    st.stop()
                
                text = extract_text_from_url(source_data)
                preprocessed_text = preprocess_text(text)
                tokenizer, model = load_pegasus_model("google/pegasus-xsum")
                summary = summarize_text(preprocessed_text, tokenizer, model, num_sentences)
                
                st.success("Summary generated successfully!")
                st.subheader("Summary:")
                st.write(summary)
                summary_displayed = True
            
            elif option == 'Text Input':
                if not source_data.strip():
                    st.error("Please enter some text")
                    st.stop()
                
                preprocessed_text = preprocess_text(source_data)
                tokenizer, model = load_pegasus_model("google/pegasus-xsum")
                summary = summarize_text(preprocessed_text, tokenizer, model, num_sentences)
                
                st.success("Summary generated successfully!")
                st.subheader("Summary:")
                st.write(summary)
                summary_displayed = True
            
            elif option == 'File Upload':
                if not source_data:
                    st.error("Please upload a valid PDF file")
                    st.stop()
                
                if summary_type == 'Whole File Summary':
                    tokenizer, model = load_pegasus_model("google/pegasus-large")
                    pages = extract_text_from_pdf(source_data)
                    full_text = ' '.join(pages)
                    preprocessed_text = preprocess_text(full_text)
                    full_summary = summarize_text(preprocessed_text, tokenizer, model, num_sentences)
                    
                    st.success("Summary generated successfully!")
                    st.subheader("Whole File Summary:")
                    st.write(full_summary)
                    summary_displayed = True

                elif summary_type == 'Page-wise Summary':
                    num_pages = st.sidebar.slider("Select number of pages to summarize:", 1, len(pages), len(pages))
            

                    if num_pages > 0:
                        st.info("Summarization in progress...")
                        tokenizer, model = load_pegasus_model("google/pegasus-large")
                        progress_text = st.empty()  # Create a placeholder for progress updates
                        
                        for i, page_text in enumerate(pages[:num_pages], 1):
                            preprocessed_text = preprocess_text(page_text)
                            summary = summarize_text(preprocessed_text, tokenizer, model, num_sentences)
                            
                            # Update progress
                            progress_text.info(f"Generated summaries for {i} out of {num_pages} pages...")
                            
                            st.subheader(f"Summary of Page {i}:")
                            st.write(summary)
                            st.markdown("---")
                        
                        # Clear the progress message and show final success message
                        progress_text.empty()
                        st.success("All summaries generated successfully!")
                        summary_displayed = True

            if not summary_displayed:
                st.warning("No summary was generated. Please check your input.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
