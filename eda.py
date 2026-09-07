import pandas as pd
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


# Load API key
load_dotenv()


# Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0
)


# Load dataset
df = pd.read_csv("data.csv")


# Get basic information
info = f"""
Shape:
{df.shape}

Columns:
{df.columns.tolist()}

Data Types:
{df.dtypes}

Missing Values:
{df.isnull().sum()}

Duplicate Rows:
{df.duplicated().sum()}

Statistics:
{df.describe().to_string()}
"""


# Prompt
prompt = ChatPromptTemplate.from_template("""
You are a Data Analyst.

Analyze this dataset:

{info}

Perform EDA and explain:

1. Dataset overview
2. Missing values
3. Duplicate values
4. Numerical columns
5. Categorical columns
6. Important statistics
7. Possible outliers
8. Important relationships
9. Recommended charts
10. Data cleaning suggestions

Explain everything in very simple language.
""")


# Create chain
chain = prompt | llm


# Ask Gemini
result = chain.invoke({
    "info": info
})


# Print result
print(result.content)
