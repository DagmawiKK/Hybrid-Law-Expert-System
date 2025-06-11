from backend.utils.logger import setup_logger

class LogisticClassifier:
    def __init__(self, gemini_api):
        self.gemini_api = gemini_api
        self.logger = setup_logger()
        # Classification prompt with examples
        self.classification_prompt = """
        You are a classifier for a legal expert system. Your task is to classify a user query as either 'symbolic' (requiring legal reasoning, deduction, or inference, such as questions about contract validity or breaches) or 'sub-symbolic' (general, descriptive, or ambiguous questions about contract law, such as definitions or explanations). Respond only with 'symbolic' or 'sub-symbolic'.

        Examples:
        - Query: "Is Contract1 valid given Alice signed it?" -> symbolic
        - Query: "What is a contract?" -> sub-symbolic
        - Query: "Does Bob owe damages for breaching Contract1?" -> symbolic
        - Query: "Explain contract law" -> sub-symbolic
        - Query: "What is quantum physics?" -> sub-symbolic

        Query: {query}
        """

    def classify(self, query):
        try:
            # Use Gemini API to classify the query
            response = self.gemini_api.llm.invoke(self.classification_prompt.format(query=query))
            classification = response.strip().lower()
            if classification not in ['symbolic', 'sub-symbolic']:
                self.logger.warning(f"Invalid classification response: {classification}, defaulting to sub-symbolic")
                return 0.5, False  # Low confidence, default to sub-symbolic
            is_symbolic = classification == 'symbolic'
            confidence = 0.9 if classification in ['symbolic', 'sub-symbolic'] else 0.5  # High confidence for valid response
            self.logger.info(f"Classified query '{query}' as {classification}")
            return confidence, is_symbolic
        except Exception as e:
            self.logger.error(f"Error classifying query: {str(e)}")
            return 0.0, False