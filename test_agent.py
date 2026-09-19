from ingest import load_record
from agent import build_agent

text, retriever = load_record("sample_lab.txt")
agent = build_agent(retriever)

out = agent.invoke({"messages": [
    {"role": "user", "content": "What is my TSH and what does it mean?"}]})

for m in out["messages"]:
    print(type(m).__name__, "|", getattr(m, "tool_calls", None) or m.content[:150])
print("\nFINAL ANSWER:\n", out["messages"][-1].content)