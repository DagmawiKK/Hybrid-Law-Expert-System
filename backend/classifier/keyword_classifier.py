from backend.utils.logger import setup_logger

class KeywordClassifier:
    def __init__(self):
        self.logger = setup_logger()
        self.symbolic_keywords = [
            'contract', 'breach', 'valid', 'invalid', 'damages', 'obligation',
            'sign', 'offer', 'acceptance', 'consideration', 'capacity', 'legal'
        ]
        self.subsymbolic_keywords = ['what', 'explain', 'describe', 'define']

    def classify(self, query):
        query_lower = query.lower()
        symbolic_score = sum(1 for keyword in self.symbolic_keywords if keyword in query_lower)
        subsymbolic_score = sum(1 for keyword in self.subsymbolic_keywords if keyword in query_lower)
        is_symbolic = symbolic_score > subsymbolic_score
        self.logger.info(f"Keyword classification: symbolic_score={symbolic_score}, subsymbolic_score={subsymbolic_score}")
        return is_symbolic