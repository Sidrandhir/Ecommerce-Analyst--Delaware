"""
main.py — CLI Entry Point
==========================
Interactive REPL for the E-Commerce AI Analyst.

Usage:
    python main.py                        # interactive mode
    python main.py --query "Top products" # single query mode
    python main.py --ingest               # re-ingest data
"""
import warnings
import argparse
import sys

# Suppress transitional SDK deprecation warnings (google-generativeai → google-genai)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_google_genai")
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from src.agents.orchestrator import EcommerceAnalystOrchestrator

console = Console()

BANNER = """
╔══════════════════════════════════════════════════════════╗
║         🛒  E-Commerce AI Analyst                        ║
║         Powered by Gemini 2.0 Flash + LangChain          ║
║         Dataset: Online Retail II (UCI)                  ║
╚══════════════════════════════════════════════════════════╝
Type your business question, or:
  'exit' / 'quit'  — exit
  'help'           — example queries
  'clear cache'    — clear response cache
"""

EXAMPLE_QUERIES = [
    "Which country generates the most revenue?",
    "What are the top 5 best-selling products by revenue?",
    "How did revenue trend month over month?",
    "Who are the highest-value customers?",
    "Which months show the highest order volumes?",
    "Compare UK vs Germany revenue and order patterns",
    "What products should we consider discontinuing?",
    "Which customer segments are most valuable?",
]


def print_result(result: dict) -> None:
    console.print(f"\n[dim]📂 Docs retrieved: {result['docs_retrieved']} | "
                  f"Filter: {result['routing'].get('filter_type', 'none')}[/dim]\n")
    console.print(Markdown(result["answer"]))
    console.print()


def interactive_mode(orchestrator: EcommerceAnalystOrchestrator) -> None:
    console.print(BANNER, style="bold cyan")

    while True:
        try:
            question = console.input("[bold green]❯ [/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Goodbye![/yellow]")
            sys.exit(0)

        if not question:
            continue

        if question.lower() in ("exit", "quit"):
            console.print("[yellow]Goodbye![/yellow]")
            sys.exit(0)

        if question.lower() == "help":
            console.print("\n[bold]Example queries:[/bold]")
            for i, q in enumerate(EXAMPLE_QUERIES, 1):
                console.print(f"  {i}. {q}")
            console.print()
            continue

        if question.lower() == "clear cache":
            from src.utils.cache import clear_cache
            clear_cache()
            console.print("[green]Cache cleared.[/green]")
            continue

        with console.status("[bold yellow]Thinking...[/bold yellow]"):
            result = orchestrator.query(question)

        print_result(result)


def single_query_mode(orchestrator: EcommerceAnalystOrchestrator, question: str) -> None:
    with console.status("[bold yellow]Thinking...[/bold yellow]"):
        result = orchestrator.query(question)
    print_result(result)


def main():
    parser = argparse.ArgumentParser(description="E-Commerce AI Analyst")
    parser.add_argument("--query", "-q", type=str, help="Single query mode")
    parser.add_argument("--ingest", action="store_true", help="Re-ingest data before querying")
    args = parser.parse_args()

    orchestrator = EcommerceAnalystOrchestrator()

    try:
        orchestrator.setup(force_ingest=args.ingest)
    except EnvironmentError as e:
        console.print(f"[bold red]Configuration Error:[/bold red] {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        console.print(f"[bold red]Data Error:[/bold red] {e}")
        console.print("[yellow]Download the dataset from: "
                      "https://archive.ics.uci.edu/dataset/502/online+retail+ii[/yellow]")
        sys.exit(1)

    if args.query:
        single_query_mode(orchestrator, args.query)
    else:
        interactive_mode(orchestrator)


if __name__ == "__main__":
    main()