# src/judge.py
from litellm import completion
import json

class FaithfulnessJudge:
    def __init__(self, model="gpt-5"):
        self.model = model

    def evaluate(self, context: list[str], answer: str) -> dict:
        """
        Scores faithfulness on a scale of 0.0 to 1.0.
        """
        prompt = f"""
        You are an expert evaluator for a RAG system.
        Your task is to check if the generated answer is faithful to the provided context.
        
        Step 1: Break the answer into individual statements.
        Step 2: For each statement, determine if it is supported by the context.
        Step 3: Count the supported statements vs total statements.
        Step 4: Return a score between 0.0 and 1.0 (Supported / Total).
        
        <context>
        {json.dumps(context)}
        </context>

        <answer>
        {answer}
        </answer>

        Output strictly in JSON format:
        {{
            "reasoning": "string",
            "score": float
        }}
        """

        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        return json.loads(response.choices[0].message.content)

class RelevanceJudge:
    def __init__(self, model="gpt-5"):
        self.model = model

    def evaluate(self, query: str, answer: str) -> dict:
        """
        Scores relevance on a scale of 0.0 to 1.0.
        """
        prompt = f"""
        You are an expert evaluator. 
        Your task is to determine if the answer addresses the user's query.
        
        Ignore whether the answer is factually true. Focus ONLY on whether it 
        is a direct response to the question asked.
        
        <query>
        {query}
        </query>

        <answer>
        {answer}
        </answer>

        Output strictly in JSON format:
        {{
            "reasoning": "string",
            "score": float
        }}
        """
        
        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return json.loads(response.choices[0].message.content)


class ContextPrecisionJudge:
    def __init__(self, model="gpt-5"):
        self.model = model

    def evaluate(self, query: str, context: list[str]) -> dict:
        """
        Scores context precision on a scale of 0.0 to 1.0.
        """
        prompt = f"""
        You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
        Your task is to evaluate the relevance of retrieved context for answering a user query.

        Step 1: Review the user query.
        Step 2: Examine each retrieved context chunk independently.
        Step 3: For each chunk, determine whether it contains information that would help answer the query.
        Step 4: Count the number of relevant chunks versus the total number of chunks.
        Step 5: Return a score between 0.0 and 1.0 (Relevant / Total).

        A chunk is considered relevant if it directly contributes information needed to answer the query.
        Irrelevant background information or tangentially related text should be marked as not relevant.

        <query>
        {query}
        </query>

        <retrieved_context>
        {json.dumps(context)}
        </retrieved_context>

        Output strictly in JSON format:
        {{
            "reasoning": "string",
            "score": float
        }}
        """

        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )

        return json.loads(response.choices[0].message.content)