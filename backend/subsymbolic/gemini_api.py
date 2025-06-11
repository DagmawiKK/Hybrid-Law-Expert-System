from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from backend.utils.logger import setup_logger

class GeminiAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.logger = setup_logger()
        self.llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
        # RAG context: Summary of contract law facts
        self.rag_context = """
        Contract Law Context:
        - Alice and Bob signed Contract1.
        - Alice offered money as consideration; Bob offered services.
        - Both accepted the offer and have legal capacity.
        - Contract1 is written, has a legal purpose, and includes payment and delivery terms.
        - Bob breached Contract1 and owes damages; Alice performed her obligations.
        - Contract1 is a valid contract with legal intent.
        """

    def get_embedding(self, text):
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding 
        except Exception as e:
            self.logger.error(f"Error getting embedding: {str(e)}")
            return [0] * 768

    def answer_query(self, query):
        try:
            # System prompt for flexible contract law responses
            system_prompt = """
            You are a legal assistant specializing in contract law. Answer questions related to contract law, including general inquiries (e.g., definitions, explanations) and specific cases. Use the provided context to ground responses for specific queries about contracts or parties mentioned in the context. For general questions, rely on your knowledge of contract law. If the query is unrelated to contract law, respond with: "This query is outside my expertise in contract law." Use clear, concise language suitable for a legal expert system.

            Context:
            {context}
            """
            prompt = f"{system_prompt.format(context=self.rag_context)}\nUser Query: {query}"
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            self.logger.error(f"Error answering query: {str(e)}")
            return "Sorry, I couldn't process that query."