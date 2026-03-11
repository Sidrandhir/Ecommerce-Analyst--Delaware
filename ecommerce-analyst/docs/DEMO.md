# Demo Documentation

## Project Overview

This system is an AI assistant designed to analyze e-commerce data using
natural language queries.

Instead of writing SQL queries or building dashboards, users can ask
questions such as:

- Which country generates the most revenue?
- What are the best-selling products?
- How did revenue change over time?

The system retrieves relevant information from the dataset and generates
clear business insights.

---

## Dataset

Dataset used:

Online Retail II (UCI)

Key information:

- ~800,000 transaction records
- 41 countries
- 5,800+ customers
- product sales data

The dataset was processed into summarized documents such as:

- country revenue summaries
- product summaries
- monthly revenue summaries
- customer summaries

These documents are stored in a vector database for retrieval.

---

## Running the System

Step 1

Install dependencies
pip install -r requirements.txt


Step 2

Add your API key in `.env`


GOOGLE_API_KEY=your_key_here


Step 3

Run ingestion


python main.py --ingest


Step 4

Start the application


python main.py


---

## Example Queries

### Query 1


Which country generates the most revenue?


Example Output Summary

The system identifies the country with the highest total revenue and
provides supporting data and business insights.

---

### Query 2


What are the top 5 best-selling products?


Example Output Summary

The system ranks products based on revenue and explains key insights
such as product popularity and customer reach.

---

### Query 3


How did monthly revenue trend across 2011?


Example Output Summary

The system analyzes monthly revenue data and highlights trends
or missing data.

---

### Query 4


Which countries should we expand into and why?


Example Output Summary

The system evaluates revenue, customer base, and average order value
to suggest potential markets for expansion.

---

## System Output Format

Each response includes structured sections:

Answer  
Key Insights  
Data Evidence  
Confidence  
Recommended Actions

This structure helps business users quickly understand the results.

---

## What This Demonstrates

This demo shows that the system can:

- retrieve relevant business data
- reason about it
- generate clear insights
- support decision making

The hybrid retrieval and agent design allow the system to handle
complex analytical queries.