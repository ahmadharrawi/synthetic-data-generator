import io
import re
import pandas as pd
import streamlit as st
import google.generativeai as genai


st.set_page_config(
    page_title="Synthetic Data Generator",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #4A00E0 0%, #8E2DE2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #666;
        margin-bottom: 1.8rem;
    }
    .stDownloadButton button {
        width: 100%;
        background-color: #2e7d32 !important;
        color: white !important;
        border: none !important;
    }
    .stDownloadButton button:hover {
        background-color: #1b5e20 !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="Enter your Gemini API key...",
        help="Get your key at https://aistudio.google.com/app/apikey",
    )
    st.markdown("---")
    st.markdown(
        """
        ### 💡 How to use
        1. Paste your **Gemini API Key** above.
        2. Enter a description of your dataset and schema.
        3. Choose the desired number of rows (10 - 200).
        4. Click **Generate Dataset**.
        5. Preview and download the generated CSV.
        """
    )



st.markdown('<div class="main-header">🧪 Synthetic Data Generator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Quickly generate realistic synthetic datasets for prototyping, testing, and analytics using Gemini 1.5 Flash.</div>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns([3, 1], gap="medium")

with col1:
    dataset_description = st.text_area(
        "Dataset Description & Schema",
        placeholder="e.g., Medical records with Patient_ID, Age, Gender, Blood_Pressure (systolic/diastolic), BMI, and Primary_Diagnosis.",
        height=140,
        help="Describe columns, data types, domains, realistic constraints, or specific patterns you need.",
    )

with col2:
    num_rows = st.slider(
        "Number of Rows",
        min_value=10,
        max_value=200,
        value=25,
        step=5,
        help="Select between 10 and 200 rows of synthetic data.",
    )

generate_btn = st.button("🚀 Generate Dataset", type="primary", use_container_width=True)


def clean_csv_output(raw_text: str) -> str:
    """Strip markdown code fence blocks, whitespace, and non-CSV wrapper text."""
    text = raw_text.strip()
    
    text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"```$", "", text, flags=re.MULTILINE)
    
    return text.strip()



if generate_btn:
    if not api_key.strip():
        st.error("⚠️ Please provide a valid Gemini API Key in the sidebar.")
    elif not dataset_description.strip():
        st.warning("⚠️ Please provide a description for the dataset you want to generate.")
    else:
        with st.spinner("Generating synthetic data with Gemini..."):
            try:
               
                genai.configure(api_key=api_key.strip())

                
                prompt = f"""You are an expert synthetic data generator.
Your task is to generate realistic synthetic tabular data based on the following description.

Dataset Description:
{dataset_description.strip()}

Requirements:
1. Generate exactly {num_rows} rows of data.
2. Output ONLY valid CSV format, including the header row.
3. Every row must match the exact number of header columns.
4. Values containing commas or special characters must be properly escaped/quoted following standard CSV rules.
5. Provide realistic, diverse, and coherent data.
6. CRITICAL: Output ONLY the raw CSV text. Do NOT include markdown code fences (like ```csv or ```), formatting tags, introductions, or conversational text. Start immediately with the header row."""

                
                model = genai.GenerativeModel("gemini-3.8-flash")
                response = model.generate_content(prompt)

                if not response or not response.text:
                    st.error("Received an empty response from the model. Please verify your prompt or API key.")
                else:
                    
                    cleaned_csv = clean_csv_output(response.text)
                    df = pd.read_csv(io.StringIO(cleaned_csv))

                    if df.empty:
                        st.warning("Parsed dataset is empty. Please check the dataset description and try again.")
                    else:
                        st.session_state["generated_df"] = df
                        st.session_state["raw_csv"] = cleaned_csv
                        st.success(f" Successfully generated {len(df)} rows and {len(df.columns)} columns!")

            except genai.types.generation_types.BlockedPromptException:
                st.error("The prompt was blocked by safety filters. Please adjust the dataset description.")
            except pd.errors.EmptyDataError:
                st.error("Failed to parse CSV: The output did not contain valid tabular data.")
                with st.expander("View Raw Output"):
                    st.code(response.text if 'response' in locals() and hasattr(response, 'text') else "")
            except pd.errors.ParserError as pe:
                st.error(f"CSV Parsing Error: {pe}. The generated output could not be formatted properly.")
                with st.expander("View Raw Output"):
                    st.code(response.text if 'response' in locals() and hasattr(response, 'text') else "")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if "generated_df" in st.session_state:
    df = st.session_state["generated_df"]

    st.markdown("---")
    st.subheader("📊 Dataset Preview")

    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Total Rows", len(df))
    with m2:
        st.metric("Total Columns", len(df.columns))
    with m3:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB")

    
    st.dataframe(df, use_container_width=True)

    
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    col_dl, _ = st.columns([1, 3])
    with col_dl:
        st.download_button(
            label="📥 Download CSV",
            data=csv_bytes,
            file_name="synthetic_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )

    
    with st.expander("🔍 Column Details & Data Types"):
        types_df = pd.DataFrame({
            "Column": df.columns,
            "Inferred Type": [str(t) for t in df.dtypes],
            "Non-Null Count": [df[c].count() for c in df.columns],
            "Unique Values": [df[c].nunique() for c in df.columns],
        })
        st.dataframe(types_df, use_container_width=True)
