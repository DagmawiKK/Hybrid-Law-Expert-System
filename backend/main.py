from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend.classifier.logistic_classifier import LogisticClassifier
from backend.classifier.keyword_classifier import KeywordClassifier
from backend.subsymbolic.gemini_api import GeminiAPI
from backend.symbolic.metta_reasoner import MettaReasoner
from backend.utils.config import GOOGLE_API_KEY
from backend.utils.logger import setup_logger
import uuid
import os


app = FastAPI(title="Law Expert System API")
logger = setup_logger()

# Initialize components
gemini_api = GeminiAPI(api_key=GOOGLE_API_KEY)
logistic_classifier = LogisticClassifier(gemini_api=gemini_api)
keyword_classifier = KeywordClassifier()
metta_reasoner = MettaReasoner(gemini_api=gemini_api)  # Pass gemini_api here!

custom_facts_cache = []

CUSTOM_FACTS_PATH = os.path.join(os.path.dirname(metta_reasoner.kb_path), "custom_facts.metta")

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def handle_query(request: QueryRequest):
    try:
        query = request.query.strip()
        logger.info(f"Received query: {query}")

        # Custom "add new facts"
        if query.lower().startswith("add new facts"):
            lines = [line.strip() for line in query.split('>')[1:] if line.strip()]
            added_facts = []
            with open(CUSTOM_FACTS_PATH, "w") as f:
                f.write("!(bind! &kb (new-space))\n")
                for fact_text in lines:
                    fact_id = f"FACT{uuid.uuid4().hex[:8]}"
                    metta_fact = parse_natural_fact_to_metta(fact_text, gemini_api, fact_id)
                    f.write(metta_fact + "\n")
                    added_facts.append(metta_fact)
            metta_reasoner.load_kb_if_needed()
            logger.info("Facts replaced:\n" + "\n".join(added_facts))
            return {"response": "Facts replaced:\n" + "\n".join(added_facts), "source": "system"}

        # Custom "add facts" command
        if query.lower().startswith("add facts"):
            lines = [line.strip() for line in query.split('>')[1:] if line.strip()]
            added_facts = []
            # Only write header if file does not exist or is empty
            file_exists = os.path.exists(CUSTOM_FACTS_PATH)
            write_header = not file_exists or os.path.getsize(CUSTOM_FACTS_PATH) == 0
            with open(CUSTOM_FACTS_PATH, "a") as f:
                if write_header:
                    f.write("!(bind! &kb (new-space))\n")
                for fact_text in lines:
                    fact_id = f"FACT{uuid.uuid4().hex[:8]}"
                    metta_fact = parse_natural_fact_to_metta(fact_text, gemini_api, fact_id)
                    f.write(metta_fact + "\n")
                    added_facts.append(metta_fact)
            metta_reasoner.load_kb_if_needed()
            logger.info("Facts added:\n" + "\n".join(added_facts))
            return {"response": "Facts added:\n" + "\n".join(added_facts), "source": "system"}

        # Classify query
        confidence, is_symbolic = logistic_classifier.classify(query)
        if confidence < 0.7:  # Threshold for fallback
            is_symbolic = keyword_classifier.classify(query)
            logger.info(f"Using keyword classifier: {'symbolic' if is_symbolic else 'sub-symbolic'}")

        # Route to appropriate AI
        if is_symbolic:
            if "infer" in query.lower() or "induction" in query.lower():
                response = metta_reasoner.process_fcc_query(query)
            else:
                response = metta_reasoner.process_query(query)
            source = "symbolic"
            logger.info("Query routed to symbolic AI")
        else:
            response = gemini_api.answer_query(query)
            source = "sub-symbolic"
            logger.info("Query routed to sub-symbolic AI")

        return {"response": response, "source": source}
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def parse_natural_fact_to_metta(fact_text, gemini_api, fact_id):
    """
    Use Gemini to convert a natural language fact to MeTTa syntax.
    """
    prompt = f"""
You are an expert in symbolic AI for contract law.
Convert each legal fact below into MeTTa syntax for a contract law expert system.
Always use the format: !(add-atom &kb (: FACT{{fact_id}} (Evaluation <predicate> <args>...)))
Choose the correct predicate and arguments based on the meaning of the fact.
Use CamelCase for parties and contracts, and match the style of the examples.

Examples:
Input: Alice signed Contract1.
Output: !(add-atom &kb (: FACT27 (Evaluation signs_contract Alice Contract1)))

Input: Bob offers services as consideration.
Output: !(add-atom &kb (: FACT28 (Evaluation offers_consideration Bob Services)))

Input: Alice accepted the offer for Contract1.
Output: !(add-atom &kb (: FACT29 (Evaluation accepts_offer Alice Contract1)))

Input: Bob has the legal capacity to contract.
Output: !(add-atom &kb (: FACT30 (Evaluation has_capacity Bob)))

Input: Contract1 is written.
Output: !(add-atom &kb (: FACT31 (Evaluation is_written Contract1)))

Input: Contract1 meets the requirement of legal purpose.
Output: !(add-atom &kb (: FACT32 (Evaluation meets_legal_purpose Contract1)))

Input: Contract1 was made with legal intent.
Output: !(add-atom &kb (: FACT33 (Evaluation has_legal_intent Contract1)))

Input: Bob breached Contract1.
Output: !(add-atom &kb (: FACT34 (Evaluation breaches_contract Bob Contract1)))

Input: Contract1 is an agreement between Alice and Bob.
Output: !(add-atom &kb (: FACT35 (Evaluation contrat_is_between Alice Bob Contract1)))

Input: Damages are available as a legal remedy.
Output: !(add-atom &kb (: FACT36 (Evaluation remedy_available damages)))

Input: Specific performance is available as a legal remedy.
Output: !(add-atom &kb (: FACT37 (Evaluation remedy_available specific_performance)))

Input: There is a statutory requirement for a written form.
Output: !(add-atom &kb (: FACT38 (Evaluation statutory_requirement written_form)))

Input: Contract1 meets the statutory requirement of written form.
Output: !(add-atom &kb (: FACT39 (Evaluation meets_statutory_requirement Contract1 written_form)))

Input: Bob's breach of Contract1 is material.
Output: !(add-atom &kb (: FACT40 (Evaluation is_material_breach Bob Contract1)))

Input: Bob did not sign Contract2.
Output: !(add-atom &kb (: FACT41 (Evaluation did_not_sign Bob Contract2)))

Input: Alice performed her obligation to pay money under Contract1.
Output: !(add-atom &kb (: FACT42 (Evaluation performs_obligation Alice pay_money Contract1)))

Input: Bob has an obligation to perform services under Contract1.
Output: !(add-atom &kb (: FACT43 (Evaluation has_obligation Bob perform_services Contract1)))

Input: Bob failed to perform his obligation to provide services under Contract1.
Output: !(add-atom &kb (: FACT44 (Evaluation fails_to_perform Bob perform_services Contract1)))

Input: There is a condition precedent: payment received by Bob under Contract1.
Output: !(add-atom &kb (: FACT45 (Evaluation condition_precedent payment_received Bob Contract1)))

Input: The condition 'payment received' has been met.
Output: !(add-atom &kb (: FACT46 (Evaluation condition_met payment_received)))

Input: Contract2 is an agreement between Alice and Bob.
Output: !(add-atom &kb (: FACT47 (Evaluation contrat_is_between Alice Bob Contract2)))

Input: The condition 'payment received' has not been met.
Output: !(add-atom &kb (: FACT48 (Evaluation condition_not_met payment_received)))

Input: Bob performed his obligation to provide services under Contract1.
Output: !(add-atom &kb (: FACT49 (Evaluation performs_obligation Bob perform_services Contract1)))

Input: {fact_text} and {fact_id} is the unique identifier for this fact. dont add any markdown just plain text. and use small letters only except for the keyword "Evaluation" and FACT{fact_id}
Output:
"""
    metta_fact = gemini_api.llm.invoke(prompt)
    return metta_fact.strip()