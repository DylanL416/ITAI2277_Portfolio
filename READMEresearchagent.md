Smart Research Assistant Agent
=============================

This repository contains the source code for the **Smart Research Assistant**, a
simple command line tool built for the ITAI2376 capstone project.  The purpose
of the agent is to demonstrate fundamental principles of agentic AI systems,
including input processing, tool integration, memory, reasoning, reinforcement
learning and safety measures.  Although the agent runs entirely offline using
pre‑defined documents, it can easily be extended to integrate real search
engines, vector databases or large language models.

Contents
--------

* **agent.py** – Main implementation of the research assistant agent.
  Defines the `DocumentRetrievalTool`, `SummarizationTool`, `CalculationTool`
  and `ResearchAssistantAgent` classes.  The agent interacts with a small
  internal document corpus and adapts its summary length using a simple
  reinforcement learning update.
* **README.md** – You are reading it.
* **requirements.txt** – Lists the Python dependencies required by this
  project (none beyond the standard library in this minimal implementation).

Quick Start
-----------

1. Ensure you have Python 3.8 or newer installed.
2. (Optional) Create a virtual environment: `python3 -m venv venv && source venv/bin/activate`.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Note: The minimal version of the agent only depends on the Python standard
   library.  External libraries are not required unless you choose to extend
   the toolset.

4. Run the agent:

   ```bash
   python agent.py
   ```

5. Enter a research topic when prompted.  The agent will search its internal
   document collection, score the results using the CRAAP criteria, generate
   summaries and display citations.  After reviewing the output you will be
   asked to rate the summary from 0 to 1.  The agent uses this feedback to
   adjust the number of sentences in future summaries.

Extending the Agent
-------------------

This code is intentionally simple to make it easy to understand and extend.  To
add more functionality i could:

* **Add new documents** – Edit the `_documents` list in `DocumentRetrievalTool`.
  Include fields for title, URL, date, domain, type (primary/secondary) and
  content.  This will allow the agent to respond to a wider range of queries.
* **Integrate web search** – Replace the `search` method in
  `DocumentRetrievalTool` with calls to a search API such as Bing, DuckDuckGo or
  the `browser` tool provided by the course.  Parse the results and return
  document objects.
* **Improve summarization** – Swap out the naive term‑frequency summariser for
  a more sophisticated algorithm (e.g. TextRank) or call a language model
  API for abstractive summarisation.
* **Persist memory** – Currently the agent stores past interactions in memory
  during runtime.  Consider writing these entries to a database (e.g. SQLite
  or a vector store) so that the agent remembers information across sessions.
* **Refine reinforcement learning** – The `update_summary_length` method uses
  a simple proportional update.  You can experiment with other RL approaches
  such as Q‑learning or policy gradients to adjust multiple parameters based
  on richer feedback.