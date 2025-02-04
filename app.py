import os
import streamlit as st
from dotenv import load_dotenv
from utils.model_utils import query_models, query_with_file, generate_complex_query
from utils.file_utils import process_uploaded_file

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="LLM Comparison Tool",
    page_icon="📊",
    layout="wide"
)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Query Comparison", "Document/Image Analysis"])

# Main content
st.title("LLM Benchmarking and Comparison Tool 🚀")

if page == "Query Comparison":
    st.header("Compare LLMs on a Query")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_area("Enter your query:", height=100)
    with col2:
        if st.button("Generate Complex Query"):
            with st.spinner("Generating a complex query..."):
                query = generate_complex_query()
                st.session_state.query = query
                st.experimental_rerun()
    
    # Display the generated query
    if "query" in st.session_state:
        query = st.session_state.query
        st.text_area("Generated Query", value=query, height=100, disabled=True)
    
    if st.button("Generate Responses"):
        if not query.strip():
            st.error("Please enter a query.")
        else:
            with st.spinner("Running query across models..."):
                results = query_models(query)
            
            # Display results with animations
            st.subheader("Results")
            for model_name, response in results.items():
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.image(f"assets/{model_name.lower().replace(' ', '-')}.png", width=50)
                with col2:
                    st.write(f"**{model_name}**")
                    st.write(response["answer"])
                    st.metric("Relevance Score", response["score"])
                st.divider()

elif page == "Document/Image Analysis":
    st.header("Analyze Documents/Images with LLMs")
    query = st.text_area("Enter your query (optional):", height=100)
    uploaded_file = st.file_uploader("Upload a document or image", type=["pdf", "docx", "xlsx", "md", "png", "jpg", "jpeg"])
    
    if st.button("Analyze"):
        if not uploaded_file:
            st.error("Please upload a file.")
        else:
            with st.spinner("Processing file and running analysis..."):
                file_content = process_uploaded_file(uploaded_file)
                results = query_with_file(query, file_content)
            
            # Display results with animations
            st.subheader("Results")
            for model_name, response in results.items():
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.image(f"assets/{model_name.lower().replace(' ', '-')}.png", width=50)
                with col2:
                    st.write(f"**{model_name}**")
                    st.write(response["answer"])
                    st.metric("Relevance Score", response["score"])
                st.divider()