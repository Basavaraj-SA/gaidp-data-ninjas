import pandas as pd
import openai

def data_profiling(df):
    """
    Perform data profiling on a pandas DataFrame.
    Returns a dictionary containing profiling information.
    """
    profiling_info = {}

    # General Information
    profiling_info['shape'] = df.shape
    profiling_info['columns'] = list(df.columns)
    profiling_info['data_types'] = df.dtypes.to_dict()
    profiling_info['missing_values'] = df.isnull().sum().to_dict()
    profiling_info['missing_percentage'] = (df.isnull().mean() * 100).to_dict()

    # Descriptive Statistics
    profiling_info['summary_statistics'] = df.describe(include='all').to_dict()

    # Unique Values
    profiling_info['unique_values'] = {col: df[col].nunique() for col in df.columns}

    # Sample Data
    profiling_info['sample_data'] = df.head(5).to_dict(orient='records')

    return profiling_info

def prepare_prompt(profiling_info):
    """
    Prepare a prompt for the LLM based on the profiling information.
    """
    prompt = f"""
    You are a data analyst. Here is a summary of a dataset:
    - Shape: {profiling_info['shape']}
    - Columns and Data Types: {profiling_info['data_types']}
    - Missing Values: {profiling_info['missing_values']}
    - Missing Percentage: {profiling_info['missing_percentage']}
    - Unique Values: {profiling_info['unique_values']}
    - Summary Statistics: {profiling_info['summary_statistics']}
    - Sample Data: {profiling_info['sample_data']}
    
    Based on this information, provide insights about the dataset, such as potential issues, patterns, or suggestions for further analysis.
    """
    return prompt

def analyze_with_llm(prompt, api_key):
    """
    Use an LLM (e.g., OpenAI GPT) to analyze the dataset based on the prompt.
    """
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

def print_profiling_report(profiling_info):
    """
    Print the profiling report in a readable format.
    """
    print("=== Data Profiling Report ===")
    print(f"Shape: {profiling_info['shape']}")
    print("\nColumns and Data Types:")
    for col, dtype in profiling_info['data_types'].items():
        print(f"  - {col}: {dtype}")
    
    print("\nMissing Values:")
    for col, missing in profiling_info['missing_values'].items():
        print(f"  - {col}: {missing} ({profiling_info['missing_percentage'][col]:.2f}%)")
    
    print("\nUnique Values:")
    for col, unique in profiling_info['unique_values'].items():
        print(f"  - {col}: {unique}")
    
    print("\nSummary Statistics:")
    for col, stats in profiling_info['summary_statistics'].items():
        print(f"  - {col}:")
        for stat, value in stats.items():
            print(f"      {stat}: {value}")
    
    print("\nSample Data:")
    for row in profiling_info['sample_data']:
        print(row)

# Example Usage
if __name__ == "__main__":
    # Load your dataset
    file_path = "D:\example_dataset.csv"  # Replace with your dataset path
    df = pd.read_csv(file_path)

    # Perform data profiling
    profiling_info = data_profiling(df)

    # Prepare the prompt for the LLM
    prompt = prepare_prompt(profiling_info)

    # Provide your OpenAI API key
    api_key = "your_openai_api_key"  # Replace with your OpenAI API key

    # Get insights from the LLM
    insights = analyze_with_llm(prompt, api_key)

    # Print the insights
    print("=== Insights from LLM ===")
    print(insights)