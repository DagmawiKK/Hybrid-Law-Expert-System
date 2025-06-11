# Hybrid Law Expert System

A hybrid symbolic and sub-symbolic AI system for answering contract law questions. This project combines symbolic reasoning (using MeTTa and custom legal rules/facts) with sub-symbolic large language model (LLM) capabilities (Google Gemini), providing both deductive legal reasoning and flexible natural language explanations.

---

## Features

- **Symbolic AI:**  
  Uses MeTTa-based knowledge base and rules for legal reasoning, proof, and inference.
- **Sub-symbolic AI:**  
  Uses Gemini LLM for general legal questions, explanations, and fallback.
- **Dynamic Fact Management:**  
  Add or replace legal facts in the knowledge base via natural language.
- **Inductive Reasoning:**  
  Supports FCC (forward chaining curried) queries for inductive proofs.
- **Interactive Frontend:**  
  Streamlit web app for chat-based legal Q&A, showing available facts and rules.

---

## Project Structure

```
Hybrid-Law-Expert-System/
│
├── backend/
│   ├── classifier/           # Query classifiers (logistic, keyword)
│   ├── subsymbolic/          # Gemini API integration
│   ├── symbolic/             # MeTTa reasoner, rules, FCC interpreter
│   ├── utils/                # Config, logging
│   ├── main.py               # FastAPI backend server
│
├── frontend/
│   ├── app.py                # Streamlit frontend
│
├── .gitignore
├── README.md
```

---

## Setup

### Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/) or `pip`
- Google Gemini API key (for LLM and embeddings)
- [Hyperon MeTTa](https://github.com/trueagi-io/hyperon-experimental) Python bindings

### Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/Hybrid-Law-Expert-System.git
    cd Hybrid-Law-Expert-System
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up environment variables:**
    - Create a `.env` file in `backend/` with:
      ```
      GOOGLE_API_KEY=your_google_gemini_api_key
      ```

4. **Run the backend server:**
    ```bash
    cd backend
    uvicorn main:app --reload
    ```

5. **Run the frontend:**
    ```bash
    cd ../frontend
    streamlit run app.py
    ```

---

## Usage

- Open the Streamlit app in your browser (usually at [http://localhost:8501](http://localhost:8501)).
- Ask questions about contract law, e.g.:
    - "Is Contract1 valid?"
    - "What is a contract?"
    - "Did Bob breach Contract1?"
- Add new facts:
    - To **replace** all facts:  
      `add new facts > Alice signed Contract3. > Bob accepted the offer for Contract3.`
    - To **append** facts:  
      `add facts > Contract3 is written. > Alice has the legal capacity to contract.`

---

## Customization

- **Add rules:**  
  Edit `backend/symbolic/symbolic_ai.metta`.
- **Add base facts:**  
  Edit `backend/symbolic/kb.metta` or use the API to add facts dynamically.

---

## License

MIT License

---

## Acknowledgments

- [Hyperon MeTTa](https://github.com/trueagi-io/hyperon-experimental)
- [Google Gemini](https://ai.google.dev/)
- [Streamlit](https://streamlit.io/)