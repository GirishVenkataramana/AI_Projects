import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# --- Styling for production-ready UI ---
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    .stTextInput input, .stTextArea textarea {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid #3A3A3A !important;
    }
    
    .stButton>button {
        background-color: #00FFAA;
        color: #000000;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
    }
    
    h1, h2, h3 {
        color: #00FFAA !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Prompt Template for classification ---
# This prompt instructs the LLM to classify the given text.
CLASSIFICATION_PROMPT = """
You are an expert text classifier. Based on the content provided, 
classify the text into one of the appropriate categories. 
If the text does not fit any standard category, simply respond with "Other".

Text: {input_text}
Category:"""

# Create a prompt chain from the template and the local Deepseek model.
classification_prompt = ChatPromptTemplate.from_template(CLASSIFICATION_PROMPT)
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")

def classify_text(input_text: str) -> str:
    """
    Given an input text, this function invokes the classification chain
    to produce a category. It includes basic error handling for production usage.
    """
    try:
        # Prepare the prompt parameters.
        prompt_params = {"input_text": input_text}
        # Generate the classification output.
        result = classification_prompt | LANGUAGE_MODEL
        classification = result.invoke(prompt_params)
        return classification.strip()
    except Exception as e:
        st.error("An error occurred while classifying the text.")
        # Log error in production environment
        # logger.error(f"Classification error: {e}")
        return "Error"

# --- Streamlit UI Configuration ---
st.title("📘 Email/Text Classifier")
st.markdown("### Classify your dynamic input text into categories")
st.markdown("---")

# Text input area for dynamic content.
user_input_text = st.text_area(
    "Enter your email or text content for classification:",
    height=200,
    help="Paste the text or email content you want to classify."
)

# Classify button triggers the classification pipeline.
if st.button("Classify"):
    if user_input_text.strip() == "":
        st.warning("Please enter some text for classification.")
    else:
        with st.spinner("Classifying text..."):
            category = classify_text(user_input_text)
        st.success("Classification Complete!")
        st.write(f"**Category:** {category}")
