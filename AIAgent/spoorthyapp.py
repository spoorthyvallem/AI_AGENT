import os
import pandas as pd
import requests
import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv


# Load environment variables
load_dotenv()
SERP_API_KEY = os.getenv("SERP_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Set the API key for Google Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Create the model configuration for Gemini 1.5 Pro
gemini_config = {
    "temperature": 0.9,
    "top_p": 1,
    "max_output_tokens": 2048,
    "response_mime_type": "text/plain",
}

gemini_model = genai.GenerativeModel(
    model_name="gemini-1.0-pro",
    generation_config=gemini_config,
)
gemini_chat = gemini_model.start_chat(
    history=[
        {
            "role": "user",
            "parts": [
                "Generate a Comprehensive engaging blog post relevant to the given title \"Effects of Generative AI\" and keywords \"Artificial Creativity, Ethical Implications, Technology Innovation, Machine Learning Applications, AI Impact on Society\". Make sure to incorporate these keywords in the blog post. The Blog should be approximately (num_words) words in length, suitable for an online audience (Ensure the Content is Original, informative, and maintains a consistent tone throughout).",
            ],
        },
    ]
)

# Streamlit UI Setup
st.title("AI Assistant Dashboard")

# File upload section
uploaded_csv = st.file_uploader("Upload a CSV file")
google_sheet_link = st.text_input("Enter Google Sheet URL (optional)")

# Load and display data
uploaded_data = None
if uploaded_csv:
    uploaded_data = pd.read_csv(uploaded_csv)
    st.write("Uploaded CSV Data", uploaded_data.head())

# Select main column for querying
if uploaded_data is not None:
    selected_column = st.selectbox("Select the main column", uploaded_data.columns)
    user_query_template = st.text_input("Enter your query (e.g., 'Get the email of {entity}')")

# Function to perform web search using SerpAPI
def perform_web_search(query):
    params = {
        "api_key": SERP_API_KEY,
        "q": query,
    }
    response = requests.get("https://serpapi.com/search", params=params)
    if response.status_code == 200:
        return response.json().get("organic_results", [])
    return []

# Function to extract information using Google Gemini
def extract_information(prompt, context_text):
    """
    Extracts information using Google Gemini 1.5 Pro based on the given prompt and context.

    :param prompt: Prompt for the model to extract relevant data.
    :param context_text: The text data for context, e.g., search results.
    :return: Extracted information as a string.
    """
    # Send the message to the chat session and get the response
    response = gemini_chat.send_message(context_text)
    return response.text.strip()

# Processing the entities from the selected column
if st.button("Run Query") and uploaded_data is not None and user_query_template:
    extraction_results = []
    for item in uploaded_data[selected_column]:
        query = user_query_template.replace("{entity}", item)
        search_results = perform_web_search(query)
        
        # Prepare web result text for LLM
        context_snippet = " ".join([result.get("snippet", "") for result in search_results])
        formatted_prompt = f"Extract the email for {item} from the following information: {context_snippet}"
        
        # Extract the information using Gemini
        extracted_data = extract_information(formatted_prompt, context_snippet)
        extraction_results.append({"Entity": item, "Extracted Info": extracted_data})

    # Display the results
    results_dataframe = pd.DataFrame(extraction_results)
    st.write("Extracted Information")
    st.dataframe(results_dataframe)

    # Download results as CSV
    csv_data = results_dataframe.to_csv(index=False)
    st.download_button("Download CSV", csv_data, "extracted_info.csv")
