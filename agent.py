from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from tools import simplify_and_flag, find_specialist

llm = ChatOllama(model="llama3.2:3b", temperature=0)

SYSTEM =  """You are a health companion that explains a patient's OWN medical record.
Rules:
- Use simple words a 12-year-old can understand.
- ALWAYS call search_record before answering a question about the record.
- Use ONLY facts from the record: the test name, the value, and the normal range.
- Say whether a value is above or below its range. Do NOT say 'slightly' or 'severe'.
- Do NOT list symptoms or possible diseases. Do NOT diagnose.
- Say 'this may need a doctor's review' for anything out of range.
- If the answer is not in the record, say you don't know.
- Keep answers to 3 to 4 sentences."""

def build_agent(retriever):
    @tool
    def search_record(question: str) -> str:
        """Search the patient's uploaded medical record for text relevant to the question."""
        return "\n\n".join(d.page_content for d in retriever.invoke(question))

    return create_agent(llm, tools=[search_record, simplify_and_flag, find_specialist],
                        system_prompt=SYSTEM)