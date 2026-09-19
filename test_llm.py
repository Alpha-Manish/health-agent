from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2:3b", temperature=0)
print(llm.invoke("Say hello in one sentence.").content)