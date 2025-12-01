# Financial AI Analyst Pro
**A Multi-Model RAG System for Advanced Banking Report Analysis**

![Streamlit Interface](RAG%20BANK%20FINAL/results/dashboard.png)

## Project Overview
Financial AI Analyst Pro is an advanced Retrieval-Augmented Generation (RAG) system engineered to automate the extraction and analysis of complex financial documents, specifically Annual Reports and Pillar 3 Disclosures.

The primary objective of this project was to overcome the limitations of standard RAG systems when dealing with dense quantitative data and unstructured layouts typical of financial reporting. We developed a modular architecture to benchmark different combinations of embedding models and Large Language Models (LLMs), aiming to identify the optimal trade-off between accuracy, latency, and cost.

### Key Features
* **Dual-Database Architecture:** The system supports two parallel pipelines: a commercial one using Voyage AI (Finance-Specific) and an open-source one using E5-Large.
* **Query Decomposition:** Implements a multi-step reasoning module that breaks down complex comparative questions (e.g., "Compare Barclays 2023 LCR vs ING") into atomic sub-queries for more precise retrieval.
* **Hybrid Search with Reranking:** Combines dense vector retrieval with a reranking step (using Voyage Rerank-2 or a local Cross-Encoder) to filter irrelevant documents and minimize hallucinations.
* **Automated Evaluation:** Includes a benchmark suite using Google Gemini 2.0 Flash as an impartial judge to score answers based on Faithfulness and Relevance.
* **User Interface:** A functional web interface built with Streamlit for real-time interaction and analysis.

---

## Architecture & Components
The system backend is built in Python and orchestrates the following components:

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Ingestion** | pdfplumber, LangChain | Precision text & table extraction with metadata injection. |
| **Vector Store** | ChromaDB | Persistent storage for embedding vectors. |
| **Embedding** | Voyage-3 / intfloat/e5-large-v2 | Dual pipeline for semantic indexing. |
| **Reasoning** | Llama 3.3 70B / Qwen 3 32B | Query decomposition and final answer generation (via Groq). |
| **Evaluation** | Gemini 2.0 Flash | Automated auditing of RAG performance. |

---

## Benchmark Results
We conducted a comparative analysis using a ground truth dataset of 25 complex financial questions. The results highlight distinct performance characteristics for each configuration.

### Performance Summary (Avg Quality Score 0-10)
1.  **E5 + Qwen 3 32B** (Score: 9.63) - Highest raw quality in testing.
2.  **Voyage + Qwen 3 32B** (Score: 9.56)
3.  **Voyage + Llama 3.3 70B** (Score: 9.42) - Recommended for production due to lower latency.
4.  **E5 + Llama 3.3 70B** (Score: 7.23)

**Key Findings:**
While Qwen achieved the highest score due to its superior noise tolerance when handling raw data, the combination of Llama 3.3 and Voyage offered the best balance for a production environment, providing high accuracy with significantly lower latency (7 seconds vs 21 seconds).

---

## Installation and Usage

### Prerequisites
* Python 3.10 or higher
* API Keys for: Groq, Voyage (Optional for Pro mode), Google (For evaluation only)

### Setup Instructions
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/SerGOLD02/Financial-AI-Analyst.git](https://github.com/SerGOLD02/Financial-AI-Analyst.git)
    cd Financial-AI-Analyst
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up environment variables:**
    Create a `.env` file in the root directory containing your API keys:
    ```env
    GROQ_API_KEY=your_groq_key
    VOYAGE_API_KEY=your_voyage_key
    GEMINI_API_KEY=your_gemini_key
    ```

4.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```

---

## Repository Structure

| File / Folder | Description |
| :--- | :--- |
| **`app.py`** | The main executable file for the **Streamlit Interface**. Run this to launch the chatbot. |
| **`research/`** | Contains the **Jupyter Notebook** (`RAG_FINANCE_DEF.ipynb`) with the complete source code for Data Ingestion, Benchmark Tournament, and Ablation Study. |
| **`RAG BANK FINAL/`** | Local storage directory (essential for running the app):<br>• `data/`: Contains raw PDF reports.<br>• `chroma_db/`: Vector store for Voyage AI embeddings.<br>• `chroma_db_e5/`: Vector store for E5 embeddings.<br>• `results/`: Benchmark logs (CSVs) and generated charts. |
| **`requirements.txt`** | List of Python dependencies required to run the project. |
| **`.env`** | (Not included) Configuration file for API Keys (Groq, Voyage, Gemini). |

---

## Contributors
* **Mandy:** Data Engineering & Ingestion Pipeline
* **Luca:** Retrieval Engine & Logic Implementation
* **Tommaso:** Model Integration & Benchmark Execution
* **Sergio:** Deep Analysis, Visualization & UI Design
