from tools import simplify_and_flag, find_specialist

txt = open("sample_lab.txt", encoding="utf-8").read()
print(simplify_and_flag.invoke({"record_text": txt}))
print(find_specialist.invoke({"flagged_tests": "tsh, hemoglobin"}))