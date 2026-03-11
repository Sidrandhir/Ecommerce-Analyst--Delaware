# Architecture Decision Record (ADR)

## Project Title
Hybrid RAG + Agentic E-Commerce Business Analyst

## 1. Business Problem

E-commerce companies generate large amounts of transactional data every day.  
Business analysts often need to answer questions such as:

- Which countries generate the most revenue?
- What are the best-selling products?
- How does revenue change over time?
- Which markets should we expand into?

Normally this analysis requires writing SQL queries or building dashboards.  
This process can be slow for non-technical users.

The goal of this project is to build an AI system that allows users to ask
business questions in natural language and receive structured insights
directly from the dataset.

The system should retrieve relevant data, reason about it, and return clear
business insights.

For this project I used the **Online Retail II dataset** which contains
transaction records from an online store.

---

## 2. Approach Chosen

I selected **Option C: Hybrid RAG + Agentic Workflow**.

This approach combines two important ideas:

**Retrieval Augmented Generation (RAG)**  
The system retrieves relevant information from the dataset before generating an answer.

**Agentic Workflow**  
Different agents perform specific tasks such as retrieving data and reasoning about it.

This approach works well for analytical questions because the system must both:

1. Retrieve correct information
2. Interpret and summarize it

---

## 3. System Architecture

The system follows a simple pipeline:

User Query  
↓  
Orchestrator  
↓  
Retrieval Agent  
↓  
Hybrid Retriever (Vector Search + BM25)  
↓  
Relevant Documents  
↓  
Analyst Agent  
↓  
Structured Business Answer

Explanation of components:

**Orchestrator**  
Controls the workflow and coordinates the agents.

**Retrieval Agent**  
Understands the query and retrieves the most relevant documents.

**Hybrid Retriever**  
Uses both semantic search (embeddings) and keyword search (BM25).

**Analyst Agent**  
Analyzes the retrieved information and generates business insights.

---

## 4. Key Technical Decisions

### LLM Selection

Model used: **Gemini 2.0 Flash**

Reasons:
- Fast response time
- Lower cost compared to larger models
- Good reasoning ability for analytical queries

This model provides a good balance between performance and cost.

---

### Vector Database

Vector database used: **ChromaDB**

Reasons:
- Lightweight and easy to run locally
- Good integration with LangChain
- No additional infrastructure required

This makes the project easy to run on a standard laptop.

---

### Chunking Strategy

Instead of embedding raw rows from the dataset, I created **summarized documents**.

Examples of documents generated:

- country revenue summaries
- product performance summaries
- monthly revenue summaries
- top customer summaries

This improves retrieval quality because the documents contain meaningful
business context rather than raw transactional data.

---

### Framework Choice

Framework used: **LangChain**

Reasons:
- Strong support for RAG pipelines
- Easy integration with vector databases
- Built-in tools for agent workflows

LangChain simplified building the retrieval and agent architecture.

---

### Agent Design

The system uses two agents.

**Retrieval Agent**

Responsibilities:
- Understand the user query
- Generate search queries
- Retrieve relevant documents

**Analyst Agent**

Responsibilities:
- Analyze the retrieved information
- Generate insights
- Structure the final response

Separating these responsibilities keeps the system modular and easier to test.

---

## 5. Trade-offs Considered

Several design trade-offs were considered during development.

**Accuracy vs Speed**

Using Gemini Flash keeps the system fast and affordable, but larger models
could potentially produce more detailed reasoning.

**Complexity vs Simplicity**

The architecture uses only two agents to keep the system simple.
More complex multi-agent systems were avoided.

**Embedding Strategy**

Summarizing the dataset into documents reduces embedding size and improves
retrieval, but it may lose some very detailed information.

---

## 6. Production Considerations

If this system were deployed in production, the following aspects would be important:

**Caching**

Responses are cached to reduce repeated API calls and lower cost.

**Retry Logic**

API requests include retry logic to handle temporary failures.

**Environment Configuration**

API keys and configuration values are stored in environment variables.

**Containerization**

A Docker configuration is included to allow easy deployment.

---

## 7. Future Improvements

If more time were available, the following improvements could be added:

- Support for multiple datasets
- Advanced evaluation metrics
- Real-time data ingestion
- A simple web interface
- Better financial aggregation logic

---

## 8. Measuring Success in Production

In a real system, success could be measured using:

- answer accuracy
- response latency
- user satisfaction
- retrieval relevance

Metrics such as **faithfulness, relevance, and context precision**
can be used to evaluate RAG performance.

---

## 9. Conclusion

This project demonstrates how a hybrid RAG and agent-based system can
transform raw business data into conversational analytics.

The architecture is simple, modular, and easy to run locally,
while still showing realistic AI system design principles.