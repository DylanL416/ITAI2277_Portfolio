"""
agent.py
===============

This module implements a simple research assistant agent intended for a capstone
project.  The goal of the agent is to demonstrate how large language model (LLM)
concepts (such as reasoning and acting) can be combined with reinforcement
learning ideas to build a useful tool.  The agent is designed to operate on
structured input data rather than the open internet because this environment
restricts network access.  Students can extend the provided stubs to call
external APIs or integrate with services such as Azure AI Studio or LangChain.

Key features provided by this module:

* **DocumentRetrievalTool** – A simple tool that returns pre‑indexed documents
  based on a query string.  Documents are stored in a local dictionary.  Each
  document contains a title, URL, publication date, domain, type (primary or
  secondary) and the full text content.  New documents can be added by editing
  the ``_documents`` dictionary.
* **SummarizationTool** – Implements a naive summarization algorithm that
  selects the top scoring sentences from a document based on term frequency.
  The number of sentences returned is controlled by the ``summary_length``
  attribute.  Stopwords are removed using a small built‑in list.
* **CalculationTool** – A simple tool that provides a handful of text
  statistics (word count, sentence count, character count).  This serves as a
  second tool to satisfy the requirement that the agent integrate at least two
  external tools.
* **ResearchAssistantAgent** – Coordinates the tools above.  It accepts a
  user query, retrieves matching documents, scores them using the CRAAP test
  criteria (currency, relevance, accuracy, authority and purpose), generates
  summaries, and compiles a report with citations.  After presenting the
  report, it solicits a reward from the user (e.g. on a 0–1 scale) and uses a
  simple reinforcement learning update to adjust the summary length.  Over
  repeated interactions the agent adapts to the user's preferences by
  increasing or decreasing the number of sentences in its summaries.

The design of the agent roughly follows a ReAct style architecture:

1. **Input Processing** – ``parse_query`` sanitizes and validates the user
   query.  It rejects potentially harmful or unsupported requests.
2. **Memory System** – ``self.memory`` stores a history of previous queries,
   retrieved documents, scores and rewards.  This could be extended to use a
   persistent vector database for long‑term memory.
3. **Reasoning Component** – ``build_report`` contains the core logic for
   deciding which documents to include and how to structure the summary.
   Documents are scored using the CRAAP criteria described in library
   guidelines【839135678671540†L83-L145】.  The agent keeps track of its current
   ``summary_length`` parameter and updates it based on feedback.
4. **Output Generation** – ``generate_output`` assembles the summary text and
   citations into a human‑readable report.  It returns both the plain text and
   a structured representation (list of citations).

The ``main`` function provides a simple command line loop so that the agent can
be exercised interactively.  It repeatedly prompts the user for a research
topic, displays the generated report, then asks for a reward score.  The
summary length is updated after each interaction.  To exit the loop, the user
can press Ctrl+C or enter an empty string.

Limitations:
--------------
This implementation relies on a small set of hard‑coded documents to avoid
network access.  It uses a very simple summarization algorithm and a naïve
reinforcement learning update.  These are included to satisfy the capstone
requirements and demonstrate the core concepts.  In a production setting you
would replace the retrieval and summarization components with calls to
search APIs, vector stores and large language models.  Likewise, a more
sophisticated RL approach could be employed (e.g. policy gradient or Q‑learning).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple


class DocumentRetrievalTool:
    """A simple document retrieval tool.

    This tool stores a small collection of documents in memory and returns those
    whose titles or content contain the query terms.  In a real system this
    component would query a search engine or database and return relevant
    results.
    """

    @dataclass
    class Document:
        title: str
        url: str
        date: str
        domain: str
        doc_type: str
        content: str

    def __init__(self) -> None:
        # Predefined document collection.  Additional entries can be added here.
        self._documents: List[DocumentRetrievalTool.Document] = [
            self.Document(
                title="Chain‑of‑Thought Prompting Elicits Reasoning in Large Language Models",
                url="https://arxiv.org/abs/2201.11903",
                date="2023-01-10",
                domain="arxiv.org",
                doc_type="primary",
                content=(
                    "Chain‑of‑Thought (CoT) prompting is a simple technique that provides a "
                    "series of intermediate reasoning steps to large language models (LLMs). "
                    "Experiments show that CoT prompting improves performance on arithmetic, "
                    "commonsense and symbolic reasoning tasks. For example, eight CoT exemplars "
                    "on a 540B‑parameter model achieved state of the art accuracy on GSM8K【889558968840036†L66-L76】."
                ),
            ),
            self.Document(
                title="ReAct: Synergizing Reasoning and Acting in Language Models",
                url="https://arxiv.org/abs/2210.03629",
                date="2023-03-10",
                domain="arxiv.org",
                doc_type="primary",
                content=(
                    "ReAct combines reasoning traces and task‑specific actions in an interleaved "
                    "manner【909090303434876†L49-L69】. Reasoning traces help the model track and update plans, "
                    "while actions interface with external sources like knowledge bases.  On question "
                    "answering and fact verification benchmarks, ReAct outperforms baselines and reduces "
                    "hallucinations【909090303434876†L49-L69】."
                ),
            ),
            self.Document(
                title="Introduction to Reinforcement Learning",
                url="https://www.geeksforgeeks.org/machine-learning/what-is-reinforcement-learning/",
                date="2025-02-24",
                domain="geeksforgeeks.org",
                doc_type="secondary",
                content=(
                    "Reinforcement learning (RL) is a branch of machine learning where an agent learns to "
                    "make decisions through trial and error to maximize cumulative rewards. The agent interacts "
                    "with an environment, receiving feedback in the form of rewards or penalties【576025642220251†L83-L107】. "
                    "Key components include the policy, reward function and value function【576025642220251†L112-L119】. "
                    "A classic example is a robot navigating a maze by exploring, receiving feedback and "
                    "adjusting its behaviour until it finds the optimal path【576025642220251†L123-L149】."
                ),
            ),
            self.Document(
                title="Evaluating Information Sources Using the CRAAP Test",
                url="https://guides.lib.uiowa.edu/evaluating",
                date="2025-06-12",
                domain="lib.uiowa.edu",
                doc_type="secondary",
                content=(
                    "The CRAAP test outlines five criteria for evaluating sources: Currency, Relevance, Accuracy, "
                    "Authority and Purpose【839135678671540†L132-L145】. Currency refers to the timeliness of the information; "
                    "relevance considers the importance and scope; accuracy assesses reliability and correctness; "
                    "authority examines the source; and purpose asks why the information exists.  These guidelines "
                    "help researchers decide which sources to trust【839135678671540†L132-L145】."
                ),
            ),
        ]

    def search(self, query: str) -> List[Document]:
        """Return all documents containing the query terms in their title or content.

        Parameters
        ----------
        query: str
            User provided search string.  The search is case‑insensitive and splits
            the query into individual terms.

        Returns
        -------
        List[Document]
            Documents whose title or content contains all query terms.
        """
        terms = [t.strip().lower() for t in re.split(r"\W+", query) if t.strip()]
        if not terms:
            return []
        results: List[DocumentRetrievalTool.Document] = []
        for doc in self._documents:
            text = f"{doc.title} {doc.content}".lower()
            if all(term in text for term in terms):
                results.append(doc)
        return results


class SummarizationTool:
    """A naive text summarization tool based on term frequency.

    This tool splits the document into sentences and assigns a score to each
    sentence based on the frequency of its non‑stopwords.  The top
    ``summary_length`` sentences are returned in their original order.  The
    summarizer does not perform deep linguistic analysis; it serves to satisfy
    project requirements without external dependencies.
    """

    def __init__(self, summary_length: int = 2) -> None:
        self.summary_length = summary_length
        # A small list of common English stopwords.  This list can be expanded.
        self.stopwords = set(
            [
                "a",
                "an",
                "and",
                "the",
                "of",
                "to",
                "is",
                "in",
                "that",
                "with",
                "for",
                "on",
                "as",
                "it",
                "by",
                "are",
            ]
        )

    def tokenize_sentences(self, text: str) -> List[str]:
        """Split a text into sentences based on punctuation.

        Parameters
        ----------
        text: str
            Document content.

        Returns
        -------
        List[str]
            Sentences extracted from the text.
        """
        # Simple sentence splitter based on punctuation.  This could be improved.
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def tokenize_words(self, sentence: str) -> List[str]:
        """Split a sentence into words, removing punctuation.

        Parameters
        ----------
        sentence: str

        Returns
        -------
        List[str]
        """
        words = re.findall(r"\b\w+\b", sentence.lower())
        return [w for w in words if w not in self.stopwords]

    def score_sentences(self, sentences: List[str]) -> Dict[int, float]:
        """Compute a score for each sentence based on term frequency.

        Returns
        -------
        Dict[int, float]
            Mapping from sentence index to score.
        """
        freq: Dict[str, int] = {}
        for sentence in sentences:
            for word in self.tokenize_words(sentence):
                freq[word] = freq.get(word, 0) + 1
        # Normalize frequencies
        max_freq = max(freq.values()) if freq else 1
        for word in freq:
            freq[word] = freq[word] / max_freq
        sentence_scores: Dict[int, float] = {}
        for i, sentence in enumerate(sentences):
            score = 0.0
            for word in self.tokenize_words(sentence):
                score += freq.get(word, 0)
            sentence_scores[i] = score
        return sentence_scores

    def summarize(self, text: str) -> str:
        """Return a summary consisting of the top scoring sentences.

        Parameters
        ----------
        text: str
            Document to summarize.

        Returns
        -------
        str
            Concatenation of the selected sentences.
        """
        sentences = self.tokenize_sentences(text)
        if not sentences:
            return ""
        scores = self.score_sentences(sentences)
        # Select sentence indices with highest scores
        top_indices = sorted(
            scores.keys(), key=lambda i: scores[i], reverse=True
        )[: self.summary_length]
        top_indices_sorted = sorted(top_indices)
        summary_sentences = [sentences[i] for i in top_indices_sorted]
        return " ".join(summary_sentences)

    def update_summary_length(self, reward: float, alpha: float = 0.5) -> None:
        """Update the summary length using a simple reinforcement learning rule.

        The summary length is adjusted based on user feedback.  Higher rewards
        encourage longer summaries; lower rewards encourage shorter summaries.
        
        Parameters
        ----------
        reward: float
            User feedback score between 0 and 1.
        alpha: float, optional
            Learning rate.  Determines how quickly the summary length changes.
        """
        # Bound the reward to [0, 1]
        reward = max(0.0, min(1.0, reward))
        # Determine target length: map reward to a sentence count (1–4)
        target_length = 1 + int(round(reward * 3))
        # Update summary length towards target using exponential smoothing
        new_length = int(round((1 - alpha) * self.summary_length + alpha * target_length))
        # Ensure length is at least 1
        self.summary_length = max(1, new_length)


class CalculationTool:
    """A simple calculation tool that reports basic statistics about text."""

    def compute_stats(self, text: str) -> Dict[str, int]:
        """Return word, sentence and character counts for the given text."""
        words = re.findall(r"\b\w+\b", text)
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        chars = len(text)
        return {
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "char_count": chars,
        }


class ResearchAssistantAgent:
    """A simple research assistant agent that integrates multiple tools."""

    def __init__(self) -> None:
        self.retrieval_tool = DocumentRetrievalTool()
        self.summarizer = SummarizationTool(summary_length=2)
        self.calc_tool = CalculationTool()
        # Memory stores past interactions: list of (query, documents, reward)
        self.memory: List[Dict[str, object]] = []

    def parse_query(self, query: str) -> Optional[str]:
        """Validate and sanitize the user query.

        Returns None if the query is not allowed (e.g. contains unsafe content).
        """
        query = query.strip()
        if not query:
            return None
        # Reject obviously unsafe or irrelevant topics
        forbidden = {"harm", "attack", "weapon", "drug", "illegal"}
        lowered = query.lower()
        if any(term in lowered for term in forbidden):
            return None
        return query

    def score_document(self, doc: DocumentRetrievalTool.Document, query: str) -> float:
        """Compute a credibility score for a document based on the CRAAP criteria.

        The score is a weighted sum of criteria:
            - Currency: newer documents score higher.
            - Relevance: documents containing all query terms are given a bonus.
            - Accuracy and Authority: primary sources and authoritative domains (.org/.edu/.gov)
              score higher.
            - Purpose: documents describing research findings (primary) score higher than
              secondary summaries.

        Parameters
        ----------
        doc: Document
            Candidate document to score.
        query: str
            User query string.

        Returns
        -------
        float
            Credibility score between 0 and 1.
        """
        # Currency: scale by age (max 1 when current year, down to 0.3 at 5 years old)
        try:
            doc_date = datetime.strptime(doc.date, "%Y-%m-%d")
        except Exception:
            currency = 0.5
        else:
            years_old = max(0, (datetime.now().year - doc_date.year))
            currency = max(0.3, 1 - 0.1 * years_old)
        # Relevance: if all query terms are in doc title/content
        terms = [t.strip().lower() for t in re.split(r"\W+", query) if t.strip()]
        text = f"{doc.title} {doc.content}".lower()
        relevance = 1.0 if all(term in text for term in terms) else 0.5
        # Accuracy/Authority: primary sources and authoritative domains
        authority_domain = 1.0 if any(doc.domain.endswith(ext) for ext in [".org", ".edu", ".gov"]) else 0.7
        accuracy = 1.0 if doc.doc_type == "primary" else 0.8
        # Purpose: penalise promotional content (none in our dataset) – assume 1
        purpose = 1.0
        # Weighted sum (weights sum to 1)
        score = 0.25 * currency + 0.25 * relevance + 0.25 * ((accuracy + authority_domain) / 2) + 0.25 * purpose
        return min(1.0, max(0.0, score))

    def build_report(self, query: str) -> Tuple[str, List[Dict[str, str]]]:
        """Retrieve documents, summarize them and assemble a report.

        Returns the report text and a list of citations.  The citations list
        contains dictionaries with keys 'index', 'title' and 'url'.
        """
        docs = self.retrieval_tool.search(query)
        if not docs:
            return ("No documents found for the query.", [])
        # Score and sort documents
        scored_docs = [(doc, self.score_document(doc, query)) for doc in docs]
        scored_docs.sort(key=lambda pair: pair[1], reverse=True)
        # For demonstration we limit to top 2 documents
        top_docs = scored_docs[:2]
        report_lines: List[str] = []
        citations: List[Dict[str, str]] = []
        for idx, (doc, score) in enumerate(top_docs, start=1):
            summary = self.summarizer.summarize(doc.content)
            stats = self.calc_tool.compute_stats(doc.content)
            report_lines.append(f"Source {idx}: {doc.title}\n" + f"Summary: {summary}\n" + f"Statistics: {stats}\n")
            citations.append({"index": str(idx), "title": doc.title, "url": doc.url})
        report_text = "\n".join(report_lines)
        return report_text, citations

    def generate_output(self, query: str) -> str:
        """Generate the report and format citations."""
        report_text, citations = self.build_report(query)
        if not citations:
            return report_text
        citation_lines = [f"[{c['index']}] {c['title']} – {c['url']}" for c in citations]
        return report_text + "\nCitations:\n" + "\n".join(citation_lines)

    def update_from_reward(self, reward: float) -> None:
        """Update internal parameters based on the provided reward."""
        self.summarizer.update_summary_length(reward)

    def log_interaction(self, query: str, docs: List[DocumentRetrievalTool.Document], reward: float) -> None:
        """Store the interaction in memory for future reference."""
        self.memory.append({"query": query, "docs": docs, "reward": reward})


def main() -> None:
    """Simple command line interface to interact with the agent."""
    agent = ResearchAssistantAgent()
    print("Welcome to the Smart Research Assistant. Enter a topic to get started.")
    try:
        while True:
            query = input("\nEnter your research topic (or press Enter to quit): ").strip()
            if not query:
                print("Goodbye!")
                break
            validated = agent.parse_query(query)
            if not validated:
                print("Sorry, that query is not allowed. Please try another topic.")
                continue
            result = agent.generate_output(validated)
            print("\n" + result)
            # Ask for feedback
            while True:
                feedback = input("\nRate this summary from 0 (bad) to 1 (excellent): ").strip()
                try:
                    reward = float(feedback)
                    break
                except ValueError:
                    print("Please enter a number between 0 and 1.")
            agent.update_from_reward(reward)
            print(f"Thank you!  The agent will adapt its summary length to {agent.summarizer.summary_length} sentences.")
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")


if __name__ == "__main__":
    main()