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


TRANSLATE_USER_MESSAGE_PROMPT = """
Translate the following user message to English:
{user_message}

Focus on improving the user message to make it more understandable and clear in English.
The translation should be concise and accurate.

RETURN FORMAT:
- Only answer with the translated message.
- Do not include any additional information.
- The translation should be in English.
"""

HALLUCINATION_DETECTOR_PROMPT = """
INSTRUCTIONS:
Assess the quality of the response based on the retrieved documents. 
The response should only contain information from the retrieved documents.
If the response contains information not present in the documents, mark it as a hallucination.

RESPONSE:
{response}

DOCUMENTS:
{documents}

RESULT:
Is the response hallucinated? Yes/No
"""
QUERIES_GENERATOR_PROMPT = """
**Objective**: Generate 5 diverse search queries tailored to retrieve information specifically from a document about biotechnology applications in veterinary vaccine development. The user's message is:

----------------
{user_message}
----------------

**Document Context**:
The document focuses on various biotechnology techniques used in veterinary vaccine development, including but not limited to:
- Reverse genetics
- Recombinant vector technology (bacterial and viral vectors)
- Gene-deleted vaccines
- Chimeric viruses
- Subunit vaccines

**Instructions**:
1. Create queries that:
   - Focus on aspects of veterinary vaccine development discussed in the document context
   - Include specific techniques (e.g., "reverse genetics vaccines")
   - Use keywords related to animal diseases, vaccine types, or specific pathogens (if relevant to the user message)
   - Explore the benefits, limitations, or applications of these biotechnologies.

2. Format requirements:
   - Each query should be concise (≤ 15 words)
   - Use a mix of keyword phrases and question formats
   - The queries should be optimized for retrieving relevant passages from the document

3. Examples of good queries (based on document context):
   - "Reverse genetics applications in veterinary vaccines"
   - "Advantages of gene-deleted vaccines for animals"
   - "Chimeric virus vaccines for livestock diseases"
   - "Recombinant vector technology in veterinary medicine"
   - "Subunit vaccines vs live vaccines for veterinary use"
"""
