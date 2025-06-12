import json
import os
import re
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import pickle
import hashlib
from hyperon import MeTTa
from backend.utils.logger import setup_logger
from backend.symbolic.fcc_interpreter import FCCInterpreter

class MettaReasoner:
    def __init__(self, gemini_api):
        self.logger = setup_logger()
        self.gemini_api = gemini_api
        self.metta = MeTTa()
        self.kb_path = os.path.join(os.path.dirname(__file__), "kb.metta")
        self.rules_path = os.path.join(os.path.dirname(__file__), "symbolic_ai.metta")
        self.kb_loaded = False

        # Initialize embedding model and semantic templates
        self.init_embedding_system()
        self.load_kb_if_needed()

        # Enhanced prompt for parsing user queries
        self.parse_prompt = """
        You are a legal query parser that understands semantic meaning and context. Analyze the user query to identify the underlying legal intent, considering synonyms, related concepts, and contextual meaning.

        The system should recognize these semantic concepts:
        - CONTRACT VALIDITY: validity, legitimacy, legal binding, enforceability, in force
        - RESTITUTION: money recovery, compensation, refunds, repayment, getting back funds
        - BREACH: contract violations, breaking terms, non-compliance, defaults, failures
        - DAMAGES: monetary compensation, financial liability, money owed, harm compensation
        - TERMINATION: ending contracts, cancellation, dissolution, revocation
        - REMEDIES: legal relief, solutions, recourse, available options, legal actions
        - PERFORMANCE: fulfillment, completion, satisfaction of obligations

        Map the semantic intent to one of these MeTTa query types:
        1. check_valid_contract: Questions about contract validity/legitimacy
        2. check_invalid_contract: Questions about contract invalidity
        3. check_restitution: Questions about money recovery/compensation
        4. check_breach: Questions about contract violations
        5. check_damages: Questions about monetary compensation owed
        6. check_termination_rights: Questions about ending contracts
        7. check_remedy: Questions about legal relief/solutions
        8. check_enforceability: Questions about legal enforceability
        9. check_excused_performance: Questions about performance exemptions
        10. check_fully_performed: Questions about complete fulfillment
        11. check_statutory_compliance: Questions about legal compliance

        Return JSON with 'intent', 'contract' (string or null), 'party' (string or null), and 'remedy' (string or null).

        Query: {query}
        Output JSON:
        """

        self.interpret_prompt = """
        You are a legal response interpreter for a contract law expert system. Given the raw MeTTa query response and the query intent, convert the response into a detailed natural language explanation suitable for a user. The MeTTa response is a list of proof structures, each containing facts (e.g., FACT1, FACT2) and rules (e.g., valid_contract_rule) leading to a conclusion (e.g., Inheritance, Evaluation). Return a single string with:
        1. A concise summary of the conclusion.
        2. A list of all facts used in the proof(s).
        3. A list of all rules applied in the proof(s).
        4. A specific mention of the contract and party involved, if applicable.
        Avoid repetition of facts and rules across multiple proofs. Format the output clearly, using bullet points for facts and rules.

        Intent: {intent}
        MeTTa Response: {metta_response}
        Output:
        """

        # Initialize FCCInterpreter with embedding model access
        self.fcc_interpreter = FCCInterpreter(gemini_api, self.interpret_prompt, self.embedding_model)

        self.load_kb_if_needed()

    def init_embedding_system(self):
        """Initialize the Gemini embedding model and semantic templates."""
        try:
            # Initialize Gemini embeddings
            api_key = getattr(self.gemini_api, 'google_api_key', None)
            if not api_key:
                api_key = os.getenv('GOOGLE_API_KEY')

            if not api_key:
                raise ValueError("Google API key not found")

            self.embedding_model = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=api_key
            )
            self.logger.info("Gemini embedding model loaded successfully")

            # Define semantic templates for each intent
            self.intent_templates = {
                'check_valid_contract': [
                    "Is the contract valid?",
                    "is some contract valid?",
                    "Is the contract legitimate?",
                    "Is the contract legally binding?",
                    "Is the contract enforceable?",
                    "Which contracts are valid?",
                    "Which contract is valid?",
                    "What contracts are legitimate?",
                    "Is the agreement legally binding?",
                    "Can the contract be enforced?",
                    "Is the contract in force?",
                    "Does the contract have legal validity?"
                ],
                'check_invalid_contract': [
                    "Is the contract invalid?",
                    "Is the contract void?",
                    "Is the contract null?",
                    "Is the contract unenforceable?",
                    "Which contracts are invalid?",
                    "Which contract is invalid?",
                    "What contracts are void?",
                    "Is the agreement illegitimate?",
                    "Has the contract expired?",
                    "Is the contract legally void?"
                ],
                'check_restitution': [
                    "Is the party entitled to restitution?",
                    "Can the party get their money back?",
                    "Should the party receive compensation?",
                    "Is the party owed a refund?",
                    "Can the party recover their funds?",
                    "Is the party entitled to repayment?",
                    "Should money be returned to the party?",
                    "Can the party get reimbursement?",
                    "Is financial recovery possible?",
                    "Does the party deserve money back?"
                ],
                'check_breach': [
                    "Who breached the contract?",
                    "Did someone violate the contract?",
                    "Who broke the contract terms?",
                    "Is there a contract violation?",
                    "Did a party default on the contract?",
                    "Who failed to perform their obligations?",
                    "Is there non-compliance with the contract?",
                    "Did someone break the agreement?",
                    "Who violated the contract terms?",
                    "Is there a breach of contract?"
                ],
                'check_damages': [
                    "Does the party owe damages?",
                    "Is the party liable for compensation?",
                    "Must the party pay money?",
                    "Is financial compensation owed?",
                    "Does the party owe monetary damages?",
                    "Is the party responsible for payment?",
                    "Should the party pay compensation?",
                    "Is money owed for damages?",
                    "Does the party have financial liability?",
                    "Must the party compensate for harm?"
                ],
                'check_termination_rights': [
                    "Can the party terminate the contract?",
                    "Is the party able to cancel the contract?",
                    "Can the party end the agreement?",
                    "Does the party have termination rights?",
                    "Can the party dissolve the contract?",
                    "Is the party allowed to revoke the contract?",
                    "Can the party exit the agreement?",
                    "Does the party have cancellation rights?",
                    "Can the contract be terminated by the party?",
                    "Can a party terminate the agreement?",
                    "can somename terminate the contract?",
                    "can some person name terminate the some contract name?",
                    "Is the party permitted to end the contract?"
                ],
                'check_remedy': [
                    "What remedies are available?",
                    "What legal relief can be sought?",
                    "What options does the party have?",
                    "What recourse is available?",
                    "What legal actions can be taken?",
                    "What solutions are possible?",
                    "What remedies can be pursued?",
                    "What legal remedies exist?",
                    "What relief is available?",
                    "What can the party do legally?"
                ],
                'check_enforceability': [
                    "Is the contract enforceable?",
                    "Can the contract be enforced?",
                    "Is the contract legally binding?",
                    "Can legal action enforce the contract?",
                    "Is the agreement enforceable in court?",
                    "Does the contract have legal force?",
                    "Can the contract be legally enforced?",
                    "Is the contract binding and enforceable?"
                ],
                'check_excused_performance': [
                    "Is the party excused from performance?",
                    "Is the party exempt from obligations?",
                    "Is the party relieved from duties?",
                    "Does the party have performance exemption?",
                    "Is the party excused from contract duties?",
                    "Is performance waived for the party?",
                    "Is the party dispensed from obligations?"
                ],
                'check_fully_performed': [
                    "Is the contract fully performed?",
                    "Has the contract been completed?",
                    "Are all obligations fulfilled?",
                    "Is the contract satisfied?",
                    "Has performance been completed?",
                    "Are all terms fulfilled?",
                    "Is the contract done?",
                    "Has everything been performed?"
                ],
                'check_statutory_compliance': [
                    "Does the contract comply with statutes?",
                    "Is the contract in compliance with law?",
                    "Does the contract meet legal requirements?",
                    "Is the contract statutorily compliant?",
                    "Does the contract satisfy regulations?",
                    "Is the contract legally compliant?",
                    "Does the contract conform to statutes?"
                ]
            }

            # Pre-compute embeddings for all templates
            self.compute_intent_embeddings()

        except Exception as e:
            self.logger.error(f"Error initializing Gemini embedding system: {str(e)}")
            self.embedding_model = None
            self.intent_embeddings = None

    def compute_intent_embeddings(self):
        """Pre-compute embeddings for all intent templates using Gemini."""
        if not self.embedding_model:
            return

        # Create cache file path
        cache_dir = os.path.join(os.path.dirname(__file__), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "gemini_intent_embeddings.pkl")

        # Create hash of templates to check if cache is valid
        templates_hash = hashlib.md5(str(self.intent_templates).encode()).hexdigest()

        # Try to load from cache
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    if cached_data.get('hash') == templates_hash:
                        self.intent_embeddings = cached_data['embeddings']
                        self.logger.info("Loaded Gemini intent embeddings from cache")
                        return
            except Exception as e:
                self.logger.warning(f"Failed to load cached Gemini embeddings: {str(e)}")

        # Compute embeddings using Gemini
        self.logger.info("Computing Gemini intent embeddings...")
        self.intent_embeddings = {}

        try:
            for intent, templates in self.intent_templates.items():
                self.logger.debug(f"Computing embeddings for intent: {intent}")

                # Get embeddings for all templates of this intent
                embeddings = self.embedding_model.embed_documents(templates)

                # Convert to numpy arrays and compute mean
                embeddings_array = np.array(embeddings)
                mean_embedding = np.mean(embeddings_array, axis=0)

                self.intent_embeddings[intent] = mean_embedding
                self.logger.debug(f"Computed mean embedding for {intent} from {len(templates)} templates")

            # Cache the results
            try:
                cache_data = {
                    'hash': templates_hash,
                    'embeddings': self.intent_embeddings
                }
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                self.logger.info("Cached Gemini intent embeddings")
            except Exception as e:
                self.logger.warning(f"Failed to cache Gemini embeddings: {str(e)}")

        except Exception as e:
            self.logger.error(f"Error computing Gemini embeddings: {str(e)}")
            self.intent_embeddings = None

    def embedding_intent_detection(self, query: str) -> Dict:
        """Detect intent using Gemini embedding-based semantic similarity."""
        if not self.embedding_model or not self.intent_embeddings:
            return {"intent": "unrecognized", "contract": None, "party": None, "remedy": None, "confidence": 0.0}

        self.logger.info(f"Running Gemini embedding-based intent detection for: {query}")

        try:
            # Extract entities first
            contract_match = re.search(r'(contract\d*)\b', query.lower()) 
            contract = contract_match.group(1) if contract_match else None
            if contract and not re.search(r'\d', contract): 
                contract = None

            party_match = re.search(r'\b(alice|bob|party1|party2|Alice|Bob|kebe|chala)\b', query.lower())
            if not party_match:
                party_match = re.search(r'(party\d+)', query.lower())
            party = party_match.group(0).capitalize() if party_match else None

            # Get query embedding using Gemini
            query_embeddings = self.embedding_model.embed_documents([query])
            query_embedding = np.array(query_embeddings[0])

            # Calculate similarity with each intent
            similarities = {}
            for intent, intent_embedding in self.intent_embeddings.items():
                similarity = cosine_similarity([query_embedding], [intent_embedding])[0][0]
                similarities[intent] = similarity
                self.logger.debug(f"Intent '{intent}' similarity: {similarity:.4f}")

            # Get best match
            best_intent = max(similarities, key=similarities.get)
            best_similarity = similarities[best_intent]

            self.logger.info(f"Best intent: {best_intent} (similarity: {best_similarity:.4f})")

            # Set threshold for confidence - Gemini embeddings might have different similarity ranges
            threshold = 0.5 

            if best_similarity >= threshold:
                return {
                    "intent": best_intent,
                    "contract": contract,
                    "party": party,
                    "remedy": None,
                    "confidence": float(best_similarity)
                }
            else:
                return {
                    "intent": "unrecognized",
                    "contract": contract,
                    "party": party,
                    "remedy": None,
                    "confidence": float(best_similarity)
                }

        except Exception as e:
            self.logger.error(f"Error in Gemini embedding intent detection: {str(e)}")
            return {"intent": "unrecognized", "contract": None, "party": None, "remedy": None, "confidence": 0.0}

    def keyword_based_fallback(self, query: str) -> Dict:
        """Fallback to keyword-based detection if embeddings fail."""
        self.logger.info("Using keyword-based fallback")

        query_lower = query.lower().strip()
        query_words = set(query_lower.split())

        contract_match = re.search(r'(contract\d*)\b', query_lower) 
        contract = contract_match.group(1) if contract_match else None
        if contract and not re.search(r'\d', contract): 
            contract = None

        party_match = re.search(r'\b(alice|bob)\b', query_lower)
        party = party_match.group(0).capitalize() if party_match else None

        # Keyword mappings
        keyword_mappings = {
            'check_valid_contract': ['valid', 'legitimate', 'legal', 'binding', 'enforceable'],
            'check_invalid_contract': ['invalid', 'void', 'null', 'unenforceable', 'illegal'],
            'check_restitution': ['restitution', 'refund', 'money', 'compensation', 'repayment', 'recovery'],
            'check_breach': ['breach', 'violation', 'broke', 'violated', 'default'],
            'check_damages': ['damages', 'compensation', 'liable', 'owes', 'pay'],
            'check_termination_rights': ['terminate', 'cancel', 'end', 'dissolve', 'revoke'],
            'check_remedy': ['remedy', 'relief', 'solution', 'recourse', 'options'],
            'check_enforceability': ['enforceable', 'enforce', 'binding'],
            'check_excused_performance': ['excused', 'exempt', 'relieved', 'waived'],
            'check_fully_performed': ['performed', 'completed', 'fulfilled', 'satisfied'],
            'check_statutory_compliance': ['complies', 'compliance', 'statute', 'regulation']
        }

        # Score each intent based on keyword matches
        scores = {}
        for intent, keywords in keyword_mappings.items():
            score = len(query_words.intersection(set(keywords)))
            if score > 0:
                scores[intent] = score

        if scores:
            best_intent = max(scores, key=scores.get)
            return {
                "intent": best_intent,
                "contract": contract,
                "party": party,
                "remedy": None,
                "confidence": scores[best_intent] / len(query_words)
            }

        return {"intent": "unrecognized", "contract": contract, "party": party, "remedy": None, "confidence": 0.0}

    def parse_query(self, query):
        """Parse query using multiple approaches in order of preference."""
        self.logger.info(f"Parsing query: {query}")

        # Try Gemini first 
        try:
            prompt = self.parse_prompt.format(query=query)
            response = self.gemini_api.llm.invoke(prompt)
            parsed = json.loads(response.strip())

            # Refine parsed entities from LLM for consistency with MeTTa
            if parsed.get('contract') and not re.search(r'contract\d+', parsed['contract'].lower()):
                parsed['contract'] = None 

            if parsed.get('party'):
                parsed['party'] = parsed['party'].capitalize()

            if parsed.get("intent") != "unrecognized":
                parsed["confidence"] = 0.9 
                return parsed
        except Exception as e:
            self.logger.error(f"Error parsing query with Gemini: {str(e)}")

        # Try Gemini embedding-based detection
        embedding_result = self.embedding_intent_detection(query)
        if embedding_result["intent"] != "unrecognized" and embedding_result["confidence"] > 0.5:
            self.logger.info(f"Gemini embedding detection successful: {embedding_result['intent']} (confidence: {embedding_result['confidence']:.3f})")
            return embedding_result

        #  Fallback to keyword-based detection
        keyword_result = self.keyword_based_fallback(query)
        if keyword_result["intent"] != "unrecognized":
            self.logger.info(f"Keyword detection successful: {keyword_result['intent']} (confidence: {keyword_result['confidence']:.3f})")
            return keyword_result

        # Return unrecognized
        self.logger.warning(f"All parsing methods failed for query: {query}")
        return {"intent": "unrecognized", "contract": None, "party": None, "remedy": None, "confidence": 0.0}


    def interpret_metta_response(self, intent, metta_response):
        # Use Gemini to interpret MeTTa response into natural language.
        try:
            metta_response_str = json.dumps(metta_response, default=str)
            prompt = self.interpret_prompt.format(intent=intent, metta_response=metta_response_str)
            response = self.gemini_api.llm.invoke(prompt)
            return response.strip()
        except Exception as e:
            self.logger.error(f"Error interpreting MeTTa response with Gemini: {str(e)}")
            return "Cannot interpret the response."

    def process_query(self, query):
        # Transform user query into MeTTa query and interpret response.
        self.logger.info(f"Processing symbolic query: {query}")
        parsed = self.parse_query(query)
        intent = parsed.get("intent")
        contract = parsed.get("contract")
        party = parsed.get("party")
        confidence = parsed.get("confidence", 0.0)


        metta_query = None 

        # Determine the target party and contract variables for MeTTa
        target_party = party if party else "$party"
        # If contract is None, use MeTTa variable $contract. Otherwise use the extracted contract.
        target_contract = contract if contract else "$contract"

        # Map intent to MeTTa query and execute
        if intent == "check_valid_contract":
            metta_query = f"!(syn &kb (fromNumber 12) (: $prf (Inheritance {target_contract} valid_contract)))"
        elif intent == "check_invalid_contract":
            metta_query = f"!(syn &kb (fromNumber 5) (: $prf (Inheritance {target_contract} invalid_contract)))"
        elif intent == "check_restitution":
            metta_query = f"!(syn &kb (fromNumber 4) (: $prf (Evaluation entitled_to_restitution {target_party} {target_contract})))"
        elif intent == "check_breach":
            metta_query = f"!(syn &kb (fromNumber 8) (: $prf (Evaluation breaches_contract {target_party} {target_contract})))"
        elif intent == "check_damages":
            metta_query = f"!(syn &kb (fromNumber 12) (: $prf (Evaluation owes_damages {target_party} {target_contract})))"
        elif intent == "check_termination_rights":
            metta_query = f"!(syn &kb (fromNumber 12) (: $prf (Evaluation can_terminate_contract {target_party} {target_contract})))"
        elif intent == "check_remedy":
            metta_query = f"!(syn &kb (fromNumber 13) (: $prf (Evaluation entitled_to_remedy {target_party} $remedy {target_contract})))"
        elif intent == "check_enforceability":
            metta_query = f"!(syn &kb (fromNumber 13) (: $prf (Evaluation enforceable_contract {target_contract})))"
        elif intent == "check_excused_performance":
            metta_query = f"!(syn &kb (fromNumber 4) (: $prf (Evaluation excused_from_performance {target_party} {target_contract})))"
        elif intent == "check_fully_performed":
            metta_query = f"!(syn &kb (fromNumber 4) (: $prf (Evaluation contract_fully_performed {target_contract})))"
        elif intent == "check_statutory_compliance":
            metta_query = f"!(syn &kb (fromNumber 4) (: $prf (Evaluation complies_with_statute {target_contract} $requirement)))"

        if metta_query:
            # Execute MeTTa query
            try:
                self.logger.info(f"Executing MeTTa query: {metta_query}")
                result = self.metta.run(metta_query)
                self.logger.debug(f"Raw MeTTa result: {result}")
                if not result:
                    return f"No specific information found for '{query}' in the knowledge base."
                return self.interpret_metta_response(intent, result)
            except Exception as e:
                self.logger.error(f"Error executing MeTTa query: {str(e)}")
                return f"An error occurred while processing your legal question: {str(e)}"
        else:
            self.logger.warning(f"Failed to generate MeTTa query for intent: {intent}. Original query: {query}")
            return f"I understand you're asking about {intent.replace('_', ' ')}. However, I could not generate a precise query for the symbolic AI based on the details provided. Please ensure all relevant parties or contracts are mentioned if applicable. (Confidence: {confidence:.3f})"

    def process_fcc_query(self, query):
        """Process an inductive (fcc) query using FCCInterpreter."""
        # Use FCCInterpreter to get the intent and FCC query
        intent = self.fcc_interpreter.embedding_fcc_match(query)
        if not intent:
            return "Sorry, I could not semantically match your inductive inference request."
        entities = self.fcc_interpreter.extract_entities(query)
        metta_query = self.fcc_interpreter.get_fcc_query(intent, entities)
        try:
            result = self.metta.run(metta_query)
            if not result:
                return f"No specific information found for '{query}' in the knowledge base."
            return self.fcc_interpreter.interpret_fcc_response(query, result)
        except Exception as e:
            return f"An error occurred while processing your legal question: {str(e)}"


    def load_kb_if_needed(self):
        self.metta = MeTTa() 

        custom_facts_path = os.path.join(os.path.dirname(self.kb_path), "custom_facts.metta")
        use_custom = os.path.exists(custom_facts_path) and os.path.getsize(custom_facts_path) > 0

        if use_custom:
            with open(custom_facts_path) as file:
                kb_facts_str = file.read()
            self.logger.info("Loading custom facts from custom_facts.metta")
        else:
            with open(self.kb_path) as file:
                kb_facts_str = file.read()
            self.logger.info("Loading facts from kb.metta")

        with open(self.rules_path) as file:
            rules_str = file.read()

        self.metta.run(kb_facts_str)
        self.metta.run(rules_str)
        self.kb_loaded = True