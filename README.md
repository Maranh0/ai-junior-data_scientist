# AI Junior Data Scientist Agent 🤖📊

An **AI-powered Junior Data Scientist Agent** that performs **Exploratory Data Analysis (EDA)** and **baseline machine learning modeling** on a real-world **fintech bank churn dataset (~10,000 customers)**.[1]
The system uses a **fully local LLM via Ollama**, orchestrated with **LangChain’s classic ReAct-style AgentExecutor**, and is exposed through a **FastAPI backend**.

This project demonstrates how agentic LLMs can autonomously decide when to run Python data tools, train models, and explain results in natural language — **without using any paid APIs**.

***

## 🚀 Project Overview

The agent is capable of:

- Loading and inspecting the dataset  
- Performing EDA (dataset overview, numeric summaries, value counts)  
- Training a **baseline churn prediction model**  
- Explaining results in natural language  
- Serving all functionality via a REST API  

**Key Result:**  
Achieved **~0.816 accuracy** on the churn prediction task using a baseline **Logistic Regression** model on the bank churn dataset (10k rows).

***

## 🧠 Architecture Overview

```
User Query
    ↓
FastAPI (/chat endpoint)
    ↓
LangChain AgentExecutor (ReAct-style agent)
    ↓
Tool Selection (EDA / Modeling)
    ↓
Python Tools (pandas, scikit-learn)
    ↓
LLM Explanation (Local via Ollama)
```

***

## 🛠️ Tech Stack

### Language & Core Libraries

- **Python 3.x**  
- **pandas**, **numpy** – data handling and EDA

### Machine Learning

- **scikit-learn**
  - Logistic Regression  
  - Train/Test split  
  - ColumnTransformer  
  - OneHotEncoder  
  - Accuracy evaluation  

### LLM & Agent Framework

- **Ollama** – fully local inference, no paid APIs (e.g., `llama3.1:8b`)
- **LangChain**  
  - `langchain-ollama` – integration with Ollama
  - `langchain-classic` `AgentExecutor` – classic agent runtime
  - `AgentType.ZERO_SHOT_REACT_DESCRIPTION` – ReAct-style tool-using agent

### Backend & Serving

- **FastAPI** – REST API for the agent
- **Uvicorn** – ASGI server for running the API  

### Monitoring & Logging

- Custom **MetricsLogger**  
  - Request latency (`latency_sec`)  
  - Total request count (`total_requests`)  
- ReAct **Thought / Action / Observation** traces visible in logs (for debugging and explainability)

***

## 📁 Project Structure

```
.
├── data_tools/
│   ├── load_data.py        # CSV loading, ID removal, target split
│   ├── eda.py              # Dataset overview, summaries, value counts
│   └── modeling.py         # Preprocessing + baseline ML model
│
├── agent_cli.py            # LangChain agent + tool wiring (AgentExecutor)
├── api_main.py             # FastAPI application
├── metrics_logger.py       # Latency and request metrics
├── data/
│   └── bank_churn.csv      # Fintech churn dataset (local, not committed)
├── requirements.txt
└── README.md
```

***

## 🔧 Implemented Tools

The agent dynamically selects from the following tools using ReAct-style reasoning:

### EDA Tools

- **basic_overview**  
  - Dataset shape  
  - Column names and data types  
  - Missing value counts  

- **numeric_summary**  
  - `pandas.DataFrame.describe()` on numeric columns  

- **value_counts(column_name)**  
  - Distribution of categorical or target variables  
  - Example: churn vs non-churn (`Exited`)  

### Modeling Tool

- **train_baseline**  

  - Drops ID columns: `RowNumber`, `CustomerId`, `Surname`  
  - Train/Test split (80/20)  
  - ColumnTransformer:  
    - Numeric features: passthrough  
    - Categorical features: OneHotEncoding (`Geography`, `Gender`, etc.)  
  - Logistic Regression (`max_iter=1000`, convergence warnings ignored for this baseline)  
  - Returns:  
    - `accuracy`  
    - `n_train` (number of training samples)  
    - `n_test` (number of test samples)  

***

## 🌐 API Endpoints

### Health Check

`GET /health`

**Response:**

```
{
  "status": "ok"
}
```

### Chat with the Agent

`POST /chat`

**Request Body:**

```
{
  "message": "Train a baseline churn model and tell me the accuracy."
}
```

**Response:**

```
{
  "reply": "The baseline logistic regression model achieved an accuracy of 0.816 on the test set.",
  "latency_sec": 1.23,
  "total_requests": 42
}
```

### 💬 Example Queries

- `"Give me a basic overview of the dataset"`  
- `"Show numeric summary"`  
- `"What are the value counts of Exited?"`  
- `"Train a baseline churn model and tell me the accuracy"`  
- `"Explain what features might drive churn in this dataset"`  

***

## 🏆 Key Achievements

- ✅ Built a **fully local agentic data scientist** (no cloud LLM required)  
- ✅ **No paid APIs** — all inference via Ollama on local hardware[8][13]
- ✅ Integrated EDA + classical ML + LLM reasoning in one system  
- ✅ Achieved **~0.816 accuracy** on a real fintech bank churn dataset[1][4]
- ✅ Clean, production-style project structure (separated tools, agent, API, metrics)  
- ✅ Easily extensible to:
  - Streamlit / React frontend  
  - Dockerized deployment  
  - Advanced models (e.g., XGBoost, SHAP explanations)  

***

## ▶️ Running the Project

1. **Start Ollama and pull a model**

   ```
   ollama pull llama3.1:8b
   ollama run llama3.1:8b "Hello"
   ```

2. **Install dependencies**

   ```
   pip install -r requirements.txt
   ```

3. **Run the API**

   ```
   uvicorn api_main:app --reload
   ```

4. **Test the Agent**

   - Open: `http://localhost:8000/docs` and use the `/chat` endpoint, or  
   - Use `curl` / Postman:

   ```
        -X POST "http://localhost:8000/chat" ^
        -H "Content-Type: application/json" ^
        -d "{\"message\": \"Train a baseline churn model and tell me the accuracy\"}"
   ```

***

## 📌 Future Improvements

- Feature importance & SHAP-based explanations  
- Multiple model comparison (Logistic Regression vs XGBoost / Random Forest)  
- Streaming agent responses  
- Frontend dashboard (Streamlit / React)  
- Docker + cloud deployment  

***

## 👤 Author

**Shahil Sinha**  
GitHub: https://github.com/TR-3N  
LinkedIn: https://linkedin.com/in/shahil-sinha-7b1636222