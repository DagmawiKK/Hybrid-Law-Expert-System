import os
import pickle
import hashlib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from backend.utils.logger import setup_logger


class FCCInterpreter:
    def __init__(self, gemini_api, interpret_prompt, embedding_model=None):
        self.gemini_api = gemini_api
        self.interpret_prompt = interpret_prompt
        self.embedding_model = embedding_model
        self.logger = setup_logger()
        
        # FCC intent to MeTTa query mapping with example phrasings
        self.fcc_templates = {
            'infer_valid_contract': (
                "!(fcc &kb (fromNumber 12) (: $prf (Inheritance {contract} valid_contract)))",
                [
                    "infer {contract} is valid",
                    "what can you infer from {contract} being valid",
                    "show induction for {contract} validity",
                    "is {contract} a valid contract by induction",
                    "prove {contract} is valid using induction",
                    "inductive proof {contract} valid",
                    "demonstrate {contract} is valid"
                ]
            ),
            'infer_invalid_contract': (
                "!(fcc &kb (fromNumber 5) (: $prf (Inheritance {contract} invalid_contract)))",
                [
                    "infer {contract} is invalid",
                    "what can you infer from {contract} being invalid",
                    "show induction for {contract} invalidity",
                    "is {contract} an invalid contract by induction",
                    "prove {contract} is invalid using induction",
                    "inductive proof {contract} invalid",
                    "demonstrate {contract} is invalid"
                ]
            ),
            'infer_entitled_to_restitution': (
                "!(fcc &kb (fromNumber 4) (: $prf (Evaluation entitled_to_restitution {party} {contract})))",
                [
                    "infer {party} is entitled to restitution for {contract}",
                    "is {party} entitled to restitution for {contract} by induction",
                    "inductive proof {party} restitution {contract}",
                    "show induction for {party} restitution {contract}",
                    "prove {party} can get restitution for {contract}"
                ]
            ),
            'infer_breaches_contract': (
                "!(fcc &kb (fromNumber 8) (: $prf (Evaluation breaches_contract {party2} {contract})))",
                [
                    "infer {party2} breached {contract}",
                    "did {party2} breach {contract} by induction",
                    "inductive proof {party2} breach {contract}",
                    "show induction for {party2} breach {contract}",
                    "prove {party2} breached {contract} using induction"
                ]
            ),
            'infer_owes_damages': (
                "!(fcc &kb (fromNumber 12) (: $prf (Evaluation owes_damages {party2} {contract})))",
                [
                    "infer {party2} owes damages for {contract}",
                    "does {party2} owe damages for {contract} by induction",
                    "inductive proof {party2} owes damages {contract}",
                    "show induction for {party2} owes damages {contract}",
                    "prove {party2} owes damages for {contract}"
                ]
            ),
            'infer_can_terminate_contract': (
                "!(fcc &kb (fromNumber 12) (: $prf (Evaluation can_terminate_contract {party} {contract})))",
                [
                    "infer {party} can terminate {contract}",
                    "can {party} terminate {contract} by induction",
                    "inductive proof {party} can terminate {contract}",
                    "show induction for {party} termination {contract}",
                    "prove {party} can terminate {contract}"
                ]
            ),
            'infer_entitled_to_remedy': (
                "!(fcc &kb (fromNumber 13) (: $prf (Evaluation entitled_to_remedy {party} damages {contract})))",
                [
                    "infer {party} is entitled to damages for {contract}",
                    "is {party} entitled to damages for {contract} by induction",
                    "inductive proof {party} remedy damages {contract}",
                    "show induction for {party} remedy damages {contract}",
                    "prove {party} is entitled to damages for {contract}",
                    "what remedy can {party} get for {contract}",
                    "demonstrate {party}'s remedy for {contract}"
                ]
            ),
            'infer_enforceable_contract': (
                "!(fcc &kb (fromNumber 13) (: $prf (Evaluation enforceable_contract {contract})))",
                [
                    "infer {contract} is enforceable",
                    "is {contract} enforceable by induction",
                    "inductive proof {contract} enforceable",
                    "show induction for {contract} enforceability",
                    "prove {contract} is enforceable",
                    "demonstrate {contract} enforceability"
                ]
            ),
            'infer_excused_from_performance': (
                "!(fcc &kb (fromNumber 4) (: $prf (Evaluation excused_from_performance {party2} {contract})))",
                [
                    "infer {party2} is excused from performance for {contract}",
                    "is {party2} excused from performance for {contract} by induction",
                    "inductive proof {party2} excused from performance {contract}",
                    "show induction for {party2} excused from performance {contract}",
                    "prove {party2} is excused from performance for {contract}",
                    "demonstrate {party2}'s performance excuse for {contract}"
                ]
            ),
            'infer_contract_fully_performed': (
                "!(fcc &kb (fromNumber 4) (: $prf (Evaluation contract_fully_performed {contract})))",
                [
                    "infer {contract} is fully performed",
                    "is {contract} fully performed by induction",
                    "inductive proof {contract} fully performed",
                    "show induction for {contract} full performance",
                    "prove {contract} is fully performed",
                    "demonstrate {contract} complete performance"
                ]
            ),
            'infer_complies_with_statute': (
                "!(fcc &kb (fromNumber 4) (: $prf (Evaluation complies_with_statute {contract} written_form)))",
                [
                    "infer {contract} complies with written_form statute",
                    "does {contract} comply with written_form by induction",
                    "inductive proof {contract} statutory compliance",
                    "show induction for {contract} statutory compliance",
                    "prove {contract} complies with written_form",
                    "demonstrate {contract} legal compliance"
                ]
            ),
        }

        # Precompute embeddings for all templates if embedding model is available
        self.fcc_embeddings = {}
        if self.embedding_model:
            self._compute_fcc_embeddings()

    def _compute_fcc_embeddings(self):
        """Compute and cache embeddings for all FCC templates."""
        try:
            # Prepare cache directory and file
            cache_dir = os.path.join(os.path.dirname(__file__), "cache")
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, "gemini_fcc_embeddings.pkl")

            # Create hash of templates to check if cache is valid
            templates_hash = hashlib.md5(str(self.fcc_templates).encode()).hexdigest()

            # Try to load from cache
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                        if cached_data.get('hash') == templates_hash:
                            self.fcc_embeddings = cached_data['embeddings']
                            print("Loaded FCC embeddings from cache")
                            return
                except Exception as e:
                    print(f"Failed to load cached FCC embeddings: {str(e)}")

            # Compute embeddings using Gemini
            print("Computing FCC intent embeddings...")
            self.fcc_embeddings = {}
            for intent, (template, examples) in self.fcc_templates.items():
                embeddings = self.embedding_model.embed_documents(examples)
                self.fcc_embeddings[intent] = np.mean(np.array(embeddings), axis=0)

            # Cache the results
            try:
                cache_data = {
                    'hash': templates_hash,
                    'embeddings': self.fcc_embeddings
                }
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                print("Cached FCC intent embeddings")
            except Exception as e:
                print(f"Failed to cache FCC embeddings: {str(e)}")

        except Exception as e:
            print(f"Error computing FCC embeddings: {str(e)}")
            self.fcc_embeddings = {}

    def embedding_fcc_match(self, query: str):
        """Find the best FCC intent for the user query using embeddings."""
        if not self.embedding_model or not self.fcc_embeddings:
            return None

        try:
            query_embedding = np.array(self.embedding_model.embed_documents([query])[0])
            similarities = {}
            
            for intent, emb in self.fcc_embeddings.items():
                similarity = cosine_similarity([query_embedding], [emb])[0][0]
                similarities[intent] = similarity

            best_intent = max(similarities, key=similarities.get)
            best_similarity = similarities[best_intent]
            threshold = 0.5 

            if best_similarity >= threshold:
                return best_intent
            return None
        except Exception as e:
            print(f"Error in FCC embedding match: {str(e)}")
            return None

    def keyword_fcc_fallback(self, query: str):
        """Fallback keyword-based matching for FCC queries."""
        query_lower = query.lower()
        
        # Simple keyword matching for FCC intents
        keyword_mappings = {
            'infer_valid_contract': ['infer', 'valid', 'contract', 'prove valid'],
            'infer_invalid_contract': ['infer', 'invalid', 'contract', 'prove invalid'],
            'infer_entitled_to_restitution': ['infer', 'restitution', 'entitled', 'prove restitution'],
            'infer_breaches_contract': ['infer', 'breach', 'breached', 'prove breach'],
            'infer_owes_damages': ['infer', 'owes', 'damages', 'prove damages'],
            'infer_can_terminate_contract': ['infer', 'terminate', 'can terminate', 'prove terminate'],
            'infer_entitled_to_remedy': ['infer', 'remedy', 'entitled', 'prove remedy'],
            'infer_enforceable_contract': ['infer', 'enforceable', 'enforce', 'prove enforceable'],
            'infer_excused_from_performance': ['infer', 'excused', 'performance', 'prove excused'],
            'infer_contract_fully_performed': ['infer', 'fully performed', 'complete', 'prove performed'],
            'infer_complies_with_statute': ['infer', 'complies', 'statute', 'prove complies']
        }
        
        best_intent = None
        best_score = 0
        
        for intent, keywords in keyword_mappings.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > best_score:
                best_score = score
                best_intent = intent
        
        return best_intent if best_score > 1 else None  
    def get_fcc_query(self, intent, entities=None):
        if intent in self.fcc_templates:
            template, _ = self.fcc_templates[intent]
            if entities is None:
                entities = {}
            # Provide defaults if not found
            defaults = {"party": "Alice", "party2": "Bob", "contract": "contract1"}
            merged = {**defaults, **{k: v for k, v in (entities or {}).items() if v}}
            return template.format(**merged)
        return None

    def interpret_fcc_response(self, query, metta_response):
        # Use Gemini to interpret an FCC (inductive) MeTTa response into natural language.
        try:
            import json
            metta_response_str = json.dumps(metta_response, default=str)
            
            # Find which FCC intent this query matches
            intent = self.embedding_fcc_match(query)
            if not intent:
                intent = self.keyword_fcc_fallback(query)
            
            if not intent:
                return "Sorry, I could not semantically match your inductive inference request."

            metta_query = self.get_fcc_query(intent)

            prompt = (
                "You are a legal response interpreter for a contract law expert system. "
                "Given the raw MeTTa FCC (inductive) query response and the original user query, "
                "explain the inference in detail. For each proof, list and explain:\n"
                "1. The facts used\n"
                "2. The rules applied\n"
                "3. The inductive steps taken\n"
                "4. The final conclusion\n"
                "Format the output clearly, using bullet points for facts and rules.\n\n"
                "Don't mention contract1 or any other contract if with in the user query there is an already defined contract where mentioning contract is apporpriate.\n\n\n"
                f"User Query: {query}\n"
                f"FCC MeTTa Query: {metta_query}\n"
                f"MeTTa FCC Response: {metta_response_str}\n"
                "Output:"
            )
            self.logger.info(f"Interpreting FCC response for query: {query} \nwith intent: {intent} and \nMeTTa response: {metta_response_str}")
            response = self.gemini_api.llm.invoke(prompt)
            return response.strip()
        except Exception as e:
            return f"Cannot interpret the FCC response: {str(e)}"

    def is_fcc_query(self, query: str) -> bool:
        # Determine if a query is requesting inductive reasoning
        fcc_keywords = [
            'infer', 'inference', 'induction', 'inductive', 
            'prove', 'proof', 'demonstrate', 'show',
            'by induction', 'using induction', 'inductive proof'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in fcc_keywords)

    def extract_entities(self, query: str):
        # Extract contract
        contract_match = re.search(r'(contract\d+)\b', query.lower())
        contract = contract_match.group(1) if contract_match else None

        # Extract parties (support multiple)
        parties = re.findall(r'\b(alice|bob|party1|party2|kebe|chala)\b', query.lower())
        parties = [p.capitalize() for p in parties]
        party = parties[0] if parties else None
        party2 = parties[1] if len(parties) > 1 else None

        return {
            "contract": contract,
            "party": party,
            "party2": party2
        }