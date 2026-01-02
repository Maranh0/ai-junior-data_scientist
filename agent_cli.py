# agent_cli.py

from typing import List

from langchain_ollama import ChatOllama

# classic agent + tools imports
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents import initialize_agent, AgentType
from langchain_classic.tools import Tool

from data_tools.load_data import load_fintech_csv, clean_and_split_id_target
from data_tools.eda import basic_overview, numeric_summary, value_counts_for_column
from data_tools.modeling import train_baseline_logreg

CSV_PATH = "data/fintech.csv"


# ---- 1. Wrap your Python functions as Tools ----

def _py_basic_overview() -> str:
    df = load_fintech_csv(CSV_PATH)
    return str(basic_overview(df))


def _py_numeric_summary() -> str:
    df = load_fintech_csv(CSV_PATH)
    return numeric_summary(df).to_string()


def _py_value_counts(column: str) -> str:
    df = load_fintech_csv(CSV_PATH)
    return value_counts_for_column(df, column).to_string()


def _py_train_baseline() -> str:
    df = load_fintech_csv(CSV_PATH)
    X, y = clean_and_split_id_target(df, target_col="Exited")
    _, metrics = train_baseline_logreg(X, y)
    return str(metrics)


def build_tools() -> List[Tool]:
    """
    Build a list of classic LangChain Tools wrapping our DS functions.
    """
    return [
        Tool(
            name="basic_overview",
            func=lambda _: _py_basic_overview(),
            description="Get a basic overview of the fintech dataset (rows, cols, dtypes, missing).",
        ),
        Tool(
            name="numeric_summary",
            func=lambda _: _py_numeric_summary(),
            description="Get numeric summary statistics (describe()) for the fintech dataset.",
        ),
        Tool(
            name="value_counts",
            func=_py_value_counts,
            description="Get value counts for a given column in the dataset. Input: column name as a string.",
        ),
        Tool(
            name="train_baseline",
            func=lambda _: _py_train_baseline(),
            description=(
                "Train a logistic regression churn model on label 'Exited' and "
                "return metrics. Use this when the user asks for model accuracy. "
                "After calling this tool once, use the accuracy in the observation "
                "to answer the question; do NOT call it again."
            ),
        ),


    ]


# ---- 2. Build classic Agent + AgentExecutor ----

def build_agent_executor() -> AgentExecutor:
    # Local LLM via Ollama
    llm = ChatOllama(
        model="llama3.1:8b",  # use the exact model name you pulled
        temperature=0.2,
    )

    tools = build_tools()

    # Classic REACT-style agent that uses tools (zero-shot)
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  
        verbose=True,
        handle_parsing_errors=True,
    )

    return agent


def main():
    agent_executor = build_agent_executor()
    print("AI Junior Data Scientist Agent (AgentExecutor + Ollama, classic API). Type 'exit' to quit.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower().strip() in {"exit", "quit"}:
            break

        # invoke AgentExecutor; it will pick tools and answer
        result = agent_executor.invoke({"input": user_input})
        # classic executors often return a dict with 'output' key
        output = result.get("output", result)
        print("\nAgent:", output)


if __name__ == "__main__":
    main()
