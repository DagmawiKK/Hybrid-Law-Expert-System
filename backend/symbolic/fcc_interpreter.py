import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

class FCCInterpreter:
    def __init__(self, gemini_api, interpret_prompt, embedding_model=None):
        self.gemini_api = gemini_api
        self.interpret_prompt = interpret_prompt
        self.embedding_model = embedding_model
        
        # FCC intent to MeTTa query mapping with example phrasings
        self.fcc_templates = {
            'infer_valid_contract': (
                "!(fcc &kb (fromNumber 12) (: $prf (Inheritance Contract1 valid_contract)))",
                [
                    "infer contract1 is valid",
                    "what can you infer from contract1 being valid",
                    "show induction for contract1 validity",
                    "is contract1 a valid contract by induction",
                    "prove contract1 is valid using induction",
                    "inductive proof contract1 valid",
                    "demonstrate contract1 is valid"
                ]
            ),
            'infer_invalid_contract': (
                "!(fcc &kb (fromNumber 5) (: $prf (Inheritance Contract2 invalid_contract)))",
                [
                    "infer contract2 is invalid",
                    "what can you infer from contract2 being invalid",
                    "show induction for contract2 invalidity",
                    "is contract2 an invalid contract by induction",
                    "prove contract2 is invalid using induction",
                    "inductive proof contract2 invalid",
                    "demonstrate contract2 is invalid"
                ]
            ),
            'infer_entitled_to_restitution': (
                "!(fcc &kb (fromNumber 4) (: $prf (Evaluation entitled_to_restitution Alice Contract2)))",
                [
                    "infer alice is entitled to restitution for contract2",
                    "is alice entitled to restitution for contract2 by induction",
                    "inductive proof alice restitution contract2",
                    "show induction for alice restitution contract2",
                    "prove alice can get restitution for contract2"
                ]
            ),
            'infer_breaches_contract': (
                "!(fcc &kb (fromNumber 8) (: $prf (Evaluation breaches_contract Bob Contract1)))",
                [
                    "infer bob breached contract1",
                    "did bob breach contract1 by induction",
                    "inductive proof bob breach contract1",
                    "show induction for bob breach contract1",
                    "prove bob breached contract1 using induction"
                ]
            ),
            'infer_owes_damages': (
                "!(fcc &kb (fromNumber 12) (: $prf (Evaluation owes_damages Bob Contract1)))",
                [
                    "infer bob owes damages for contract1",
                    "does bob owe damages for contract1 by induction",
                    "inductive proof bob owes damages contract1",
                    "show induction for bob owes damages contract1",
                    "prove bob owes damages for contract1"
                ]
            ),
            'infer_can_terminate_contract': (
                "!(fcc &kb (fromNumber 12) (: $prf (Evaluation can_terminate_contract Alice Contract1)))",
                [
                    "infer alice can terminate contract1",
                    "can alice terminate contract1 by induction",
                    "inductive proof alice can terminate contract1",
                    "show induction for alice termination contract1",
                    "prove alice can terminate contract1"
                ]
            ),
            'infer_entitled_to_remedy': (
                "!(fcc &kb (fromNumber 13) (: $prf (Evaluation entitled_to_remedy Alice damages Contract1)))",
                [
                    "infer alice is entitled to damages for contract1",
                    "is alice entitled to damages for contract1 by induction",
                    "inductive proof alice remedy damages contract1",
                    "show induction for alice remedy damages contract1",
                    "prove alice is entitled to damages for contract1",
                    "what remedy can alice get for contract1",
                    "demonstrate alice's remedy for contract1"
                ]
            ),
            'infer_enforceable_contract': (
                "!(fcc &kb (fromNumber 13) (: $prf (Evaluation enforceable_contract Contract1)))",
                [
                    "infer contract1 is enforceable",
                    "is contract1 enforceable by induction",
                    "inductive proof contract1 enforceable",
                    "show induction for contract1 enforceability",
                    "prove contract1 is enforceable",
                    "demonstrate contract1 enforceability"
                ]
            ),
            'infer_excused_from_performance': (
                "!(fcc &kb (fromNumber 4) (: $prf (Evaluation excused_from_performance Bob Contract1)))",
                [
                    "infer bob is excused from performance for contract1",
                    "is bob excused from performance for contract1 by induction",
                    "inductive proof bob excused from performance contract1",
                    "show induction for bob excused from performance contract1",
                    "prove bob is excused from performance for contract1",
                    "demonstrate bob's performance excuse for contract1"
                ]
            ),
            'infer_contract_fully_performed': (
                "!(fcc &kb (fromNumber 4) (: $prf (Evaluation contract_fully_performed Contract1)))",
                [
                    "infer contract1 is fully performed",
                    "is contract1 fully performed by induction",
                    "inductive proof contract1 fully performed",
                    "show induction for contract1 full performance",
                    "prove contract1 is fully performed",
                    "demonstrate contract1 complete performance"
                ]
            ),
            'infer_complies_with_statute': (
                "!(fcc &kb (fromNumber 4) (: $prf (Evaluation complies_with_statute Contract1 written_form)))",
                [
                    "infer contract1 complies with written_form statute",
                    "does contract1 comply with written_form by induction",
                    "inductive proof contract1 statutory compliance",
                    "show induction for contract1 statutory compliance",
                    "prove contract1 complies with written_form",
                    "demonstrate contract1 legal compliance"
                ]
            ),
        }

        # Precompute embeddings for all templates if embedding model is available
        self.fcc_embeddings = {}
        if self.embedding_model:
            self._compute_fcc_embeddings()

    def _compute_fcc_embeddings(self):
        """Compute embeddings for all FCC templates."""
        try:
            for intent, (template, examples) in self.fcc_templates.items():
                embeddings = self.embedding_model.embed_documents(examples)
                self.fcc_embeddings[intent] = np.mean(np.array(embeddings), axis=0)
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
    def get_fcc_query(self, intent):
        # Return the exact FCC query for the matched intent
        if intent in self.fcc_templates:
            template, _ = self.fcc_templates[intent]
            return template
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
                f"User Query: {query}\n"
                f"FCC MeTTa Query: {metta_query}\n"
                f"MeTTa FCC Response: {metta_response_str}\n"
                "Output:"
            )
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