"""
Ingestion Pipeline — Online Retail II (UCI)
"""
import os
import warnings
import pandas as pd
from pathlib import Path
from typing import List

warnings.filterwarnings("ignore", category=FutureWarning, module="langchain_google_genai")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_google_genai")

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import (
    DATA_PATH, CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP, GOOGLE_API_KEY, validate
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Confirmed working embedding model for this API key
EMBEDDING_MODEL = "models/gemini-embedding-001"


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    logger.info(f"Loading dataset from {path}")
    df = pd.read_excel(path, sheet_name=None)

    frames = []
    for sheet_name, frame in df.items():
        frame["_sheet"] = sheet_name
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)

    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    required = ["Invoice", "StockCode", "Description", "Quantity",
                "InvoiceDate", "Price", "Customer_ID", "Country"]
    available = [c for c in required if c in df.columns]
    df = df.dropna(subset=available)

    df = df[~df["Invoice"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    df["LineTotal"] = df["Quantity"] * df["Price"]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    df["Year"] = df["InvoiceDate"].dt.year

    logger.info(f"Dataset loaded: {len(df):,} rows, {df['Customer_ID'].nunique():,} customers, "
                f"{df['Country'].nunique()} countries")
    return df


def build_documents(df: pd.DataFrame) -> List[Document]:
    docs: List[Document] = []

    logger.info("Building product summaries...")
    prod_grp = (
        df.groupby(["StockCode", "Description"])
        .agg(
            total_revenue=("LineTotal", "sum"),
            total_units=("Quantity", "sum"),
            num_orders=("Invoice", "nunique"),
            num_customers=("Customer_ID", "nunique"),
            avg_unit_price=("Price", "mean"),
            countries_sold=("Country", "nunique"),
        )
        .reset_index()
        .sort_values("total_revenue", ascending=False)
        .head(500)
    )
    for _, row in prod_grp.iterrows():
        text = (
            f"Product: {row['Description']} (StockCode: {row['StockCode']})\n"
            f"Total Revenue: £{row['total_revenue']:,.2f}\n"
            f"Units Sold: {int(row['total_units']):,}\n"
            f"Number of Orders: {int(row['num_orders']):,}\n"
            f"Unique Customers: {int(row['num_customers']):,}\n"
            f"Average Unit Price: £{row['avg_unit_price']:.2f}\n"
            f"Countries Sold In: {int(row['countries_sold'])}\n"
        )
        docs.append(Document(
            page_content=text,
            metadata={"type": "product", "stock_code": str(row["StockCode"]),
                      "description": str(row["Description"])}
        ))

    logger.info("Building country summaries...")
    country_grp = (
        df.groupby("Country")
        .agg(
            total_revenue=("LineTotal", "sum"),
            num_orders=("Invoice", "nunique"),
            num_customers=("Customer_ID", "nunique"),
            avg_order_value=("LineTotal", lambda x: x.groupby(
                df.loc[x.index, "Invoice"]).sum().mean()),
        )
        .reset_index()
        .sort_values("total_revenue", ascending=False)
    )
    for _, row in country_grp.iterrows():
        text = (
            f"Country: {row['Country']}\n"
            f"Total Revenue: £{row['total_revenue']:,.2f}\n"
            f"Number of Orders: {int(row['num_orders']):,}\n"
            f"Unique Customers: {int(row['num_customers']):,}\n"
            f"Average Order Value: £{row['avg_order_value']:.2f}\n"
        )
        docs.append(Document(
            page_content=text,
            metadata={"type": "country", "country": str(row["Country"])}
        ))

    logger.info("Building monthly summaries...")
    month_grp = (
        df.groupby("Month")
        .agg(
            total_revenue=("LineTotal", "sum"),
            num_orders=("Invoice", "nunique"),
            num_customers=("Customer_ID", "nunique"),
            units_sold=("Quantity", "sum"),
        )
        .reset_index()
        .sort_values("Month")
    )
    for _, row in month_grp.iterrows():
        text = (
            f"Monthly Report — {row['Month']}\n"
            f"Total Revenue: £{row['total_revenue']:,.2f}\n"
            f"Number of Orders: {int(row['num_orders']):,}\n"
            f"Unique Customers: {int(row['num_customers']):,}\n"
            f"Units Sold: {int(row['units_sold']):,}\n"
        )
        docs.append(Document(
            page_content=text,
            metadata={"type": "monthly", "month": str(row["Month"])}
        ))

    logger.info("Building top-customer summaries...")
    cust_grp = (
        df.groupby("Customer_ID")
        .agg(
            total_spend=("LineTotal", "sum"),
            num_orders=("Invoice", "nunique"),
            country=("Country", "first"),
            first_order=("InvoiceDate", "min"),
            last_order=("InvoiceDate", "max"),
        )
        .reset_index()
        .sort_values("total_spend", ascending=False)
        .head(200)
    )
    for _, row in cust_grp.iterrows():
        text = (
            f"Customer ID: {int(row['Customer_ID'])}\n"
            f"Country: {row['country']}\n"
            f"Total Spend: £{row['total_spend']:,.2f}\n"
            f"Number of Orders: {int(row['num_orders'])}\n"
            f"First Order: {str(row['first_order'])[:10]}\n"
            f"Last Order: {str(row['last_order'])[:10]}\n"
        )
        docs.append(Document(
            page_content=text,
            metadata={"type": "customer", "customer_id": str(int(row["Customer_ID"]))}
        ))

    logger.info(f"Total documents built: {len(docs)}")
    return docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"Chunks after splitting: {len(chunks)}")
    return chunks


def build_vectorstore(chunks: List[Document]) -> Chroma:
    embeddings = get_embeddings()
    logger.info(f"Embedding {len(chunks)} chunks → ChromaDB at {CHROMA_PERSIST_DIR}")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION_NAME,
    )
    logger.info("Vectorstore persisted successfully.")
    return vectorstore


def load_vectorstore() -> Chroma:
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
    )


def vectorstore_exists() -> bool:
    p = Path(CHROMA_PERSIST_DIR)
    return p.exists() and any(p.iterdir())


def run_ingestion(force: bool = False) -> Chroma:
    validate()
    if vectorstore_exists() and not force:
        logger.info("Vectorstore already exists. Loading from disk. (Use --ingest to rebuild)")
        return load_vectorstore()

    df = load_dataset()
    docs = build_documents(df)
    chunks = chunk_documents(docs)
    vs = build_vectorstore(chunks)
    return vs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_ingestion(force=args.force)
    print("Ingestion complete.")