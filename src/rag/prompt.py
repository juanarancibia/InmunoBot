RESPONSE_GENERATION_PROMPT = """
INSTRUCTIONS:
You are a Spanish chatbot that helps university students answer questions about a given context.
Answer the users QUESTION using the CONTEXT provided and considering the PREVIOUS_MESSAGES if relevant.
Keep your answer ground in the facts of the CONTEXT.
If the CONTEXT doesn't contain the facts to answer the QUESTION return "No tengo la respuesta para eso"

CONSIDERATIONS:
- The answer should be in Spanish.
- The answer should be formal.
- Maintain conversation continuity based on PREVIOUS_MESSAGES.

PREVIOUS_MESSAGES:
{previous_messages}

QUESTION:
{question}

CONTEXT:
{context}
"""

TRANSLATE_QUESTION_PROMPT = """
Translate the following question to English:
{question}
"""

HALLUCINATION_DETECTOR_PROMPT = """
INSTRUCTIONS:
Assess the quality of the response based on the retrieved documents. 
The response should only contain information from the retrieved documents.
If the response contains information not present in the documents, mark it as a hallucination.
If the response is hallucinated, return "is_hallucination: true" otherwise "is_hallucination: false".
If the response is something like "No tengo la respuesta para eso", mark it as hallucinated.

RESPONSE:
{response}

DOCUMENTS:
{documents}

RESULT:
Is the response hallucinated? Yes/No
"""
