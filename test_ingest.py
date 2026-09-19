from ingest import load_record

text, retriever = load_record("sample_lab.txt")
for q in ["What is my TSH?", "What medicine am I taking?", "When should I repeat the test?"]:
    print("\nQ:", q)
    for d in retriever.invoke(q):
        print("  -", d.metadata["section"], "|", d.page_content[:70].replace("\n", " "))