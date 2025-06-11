import streamlit as st
import requests
import os
# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# API endpoint
API_URL = "http://localhost:8000/query"

# Title and description
st.title("Law Expert System")
st.write("Ask questions about contract law. Responses are labeled as 'Symbolic' (reasoning-based) or 'Sub-symbolic' (descriptive).")

# Show Facts and Rules from symbolic_ai.metta

def extract_facts_and_rules():
    facts = []
    rules = []
    metta_path = os.path.join(os.path.dirname(__file__), "..", "backend", "symbolic", "symbolic_ai.metta")
    try:
        with open(metta_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("!(") and "add-atom" in line:
                    facts.append(interpret_fact_or_rule(line))
                elif line.startswith("!(") and "add-reduct" in line:
                    rules.append(interpret_fact_or_rule(line, False))
    except Exception as e:
        facts = [f"Error reading metta file: {e}"]
        rules = []
    return facts, rules

def interpret_fact_or_rule(line: str, is_fact=True) -> str:
    import re
    key = None
    if is_fact:
        m = re.search(r"(FACT\d+)", line)
        if m:
            key = m.group(1)
    else:
        key = line

    match key:
        # Facts
        case "FACT1":
            return "Alice signed Contract1."
        case "FACT2":
            return "Bob signed Contract1."
        case "FACT3":
            return "Alice offers money as consideration."
        case "FACT4":
            return "Bob offers services as consideration."
        case "FACT5":
            return "Alice accepted the offer for Contract1."
        case "FACT6":
            return "Bob accepted the offer for Contract1."
        case "FACT7":
            return "Alice has the legal capacity to contract."
        case "FACT8":
            return "Bob has the legal capacity to contract."
        case "FACT9":
            return "Contract1 is written."
        case "FACT10":
            return "Contract1 meets the requirement of legal purpose."
        case "FACT11":
            return "Contract1 was made with legal intent."
        case "FACT12":
            return "Bob breached Contract1."
        case "FACT13":
            return "Contract1 is an agreement between Alice and Bob."
        case "FACT14":
            return "Damages are available as a legal remedy."
        case "FACT15":
            return "Specific performance is available as a legal remedy."
        case "FACT16":
            return "There is a statutory requirement for a written form."
        case "FACT17":
            return "Contract1 meets the statutory requirement of written form."
        case "FACT18":
            return "Bob's breach of Contract1 is material."
        case "FACT19":
            return "Bob did not sign Contract2."
        case "FACT20":
            return "Alice performed her obligation to pay money under Contract1."
        case "FACT21":
            return "Bob has an obligation to perform services under Contract1."
        case "FACT22":
            return "Bob failed to perform his obligation to provide services under Contract1."
        case "FACT23":
            return "There is a condition precedent: payment received by Bob under Contract1."
        case "FACT24":
            return "The condition 'payment received' has been met."
        case "FACT25":
            return "Contract2 is an agreement between Alice and Bob."
        case "FACT26":
            return "The condition 'payment received' has not been met."
        case "FACT27":
            return "Bob performed his obligation to provide services under Contract1."
        # Rules
        case "valid_contract_rule":
            return "A contract is valid if all legal requirements are met."
        case "owes_damages_rule":
            return "A party owes damages if they breach a valid contract."
        case "breach_rule":
            return "A breach occurs when a party fails to fulfill their contractual obligations."
        case "termination_option_rule":
            return "A contract may be terminated under specific conditions, such as a material breach."
        case "invalid_contract_rule":
            return "A contract is invalid if it fails to meet legal requirements."
        case "restitution_rule":
            return "A party may be entitled to restitution if certain conditions are met."
        case "remedy_rule":
            return "Legal remedies are available when a contract is breached."
        case "statutory_compliance_rule":
            return "A contract must comply with relevant statutes to be valid."
        case "condition_precedent_rule":
            return "Performance is excused if a condition precedent is not met."
        case "fully_performed_rule":
            return "A contract is fully performed when all obligations are fulfilled."
        case "enforceability_rule":
            return "A contract is enforceable if it is valid and complies with statutory requirements."
        case _:
            return "No plain English interpretation available."


with st.expander("🔎 Show available facts and rules in Symbolic AI"):
    facts, rules = extract_facts_and_rules()
    st.subheader("Facts")
    for i, fact in enumerate(facts, 1):
        st.code(fact, language="metta")
        st.write(f"**Fact{i}**")
    st.subheader("Rules")
    rules = [
        "valid_contract_rule",
        "owes_damages_rule",
        "breach_rule",
        "termination_option_rule",
        "invalid_contract_rule",
        "restitution_rule",
        "remedy_rule",
        "statutory_compliance_rule",
        "condition_precedent_rule",
        "fully_performed_rule",
        "enforceability_rule"
    ]
    for i, rule in enumerate(rules):
        st.code(interpret_fact_or_rule(rule, False), language="metta")
        import re
        m = re.search(r"([a-z_]+_rule)", rule)
        label = m.group(1) if m else "Rule"
        st.write(f"**{label.replace('_', ' ').capitalize()}**")

# Chat container
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            st.markdown(f"_{message['source']} AI_")

# Chat input
if prompt := st.chat_input("Enter your query (e.g., Is Contract1 valid? or What is a contract?)"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt, "source": None})
    
    # Show loading spinner
    with st.spinner("Processing your query..."):
        try:
            # Send query to FastAPI
            response = requests.post(API_URL, json={"query": prompt})
            response.raise_for_status()
            result = response.json()
            answer = result.get("response", "No response received")
            source = result.get("source", "Unknown")
            
            # Add assistant response with source label
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "source": source.capitalize()
            })
            
            # Rerun to update UI
            st.rerun()
        except requests.RequestException as e:
            st.error(f"Error: {str(e)}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Error: {str(e)}",
                "source": "Error"
            })
            st.rerun()