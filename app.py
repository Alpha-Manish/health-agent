import json, html
import gradio as gr
from ingest import load_record
from agent import build_agent
from tools import simplify_and_flag, find_specialist

DISCLAIMER = ("This is a simplified explanation, not a diagnosis or medical advice. "
              "Please consult a doctor.")

CSS = """
.hero {text-align:center; padding:24px; border-radius:16px;
       background:linear-gradient(135deg,#0ea5e9,#6366f1); color:white;}
.hero h1 {color:white; margin:0;}
.card {padding:12px 16px; border-radius:12px; margin:8px 0; border-left:6px solid;}
.ok     {background:#ecfdf5; border-color:#10b981; color:#065f46;}
.flag   {background:#fef2f2; border-color:#ef4444; color:#7f1d1d;}
.follow {background:#fffbeb; border-color:#f59e0b; color:#78350f;}
.badge {font-weight:700; margin-right:8px;}
.disc {background:#f1f5f9; color:#334155; padding:10px 14px; border-radius:8px; font-size:0.9em;}
"""

def to_cards(lines):
    if not lines:
        return ('<div class="card follow">No known lab values were found in this document. '
                'You can still ask questions below.</div>')
    out = ""
    for l in lines:
        if l.startswith("[OK]"):
            cls, tag, text = "ok", "NORMAL", l[4:]
        elif l.startswith("[FLAG]"):
            cls, tag, text = "flag", "NEEDS ATTENTION", l[6:]
        else:
            cls, tag, text = "follow", "FOLLOW-UP", l.replace("[FOLLOW-UP]", "")
        out += f'<div class="card {cls}"><span class="badge">{tag}</span>{html.escape(text.strip())}</div>'
    return out

def on_upload(file, state):
    if not file:
        return "", "", [], {}
    text, retriever = load_record(file)                                   # 1. ingest
    agent = build_agent(retriever)
    res = json.loads(simplify_and_flag.invoke({"record_text": text}))     # 2. tool 1, automatic
    spec = find_specialist.invoke({"flagged_tests": ", ".join(res["flagged_tests"])})  # 3. tool 2
    who = "### Who to see\n" + "\n".join(f"- {l}" for l in spec.splitlines())
    state = {"agent": agent, "summary": "\n".join(res["summary"]) + "\n" + spec}
    return to_cards(res["summary"]), who, [], state

def on_chat(msg, history, state):
    history = history or []
    if not msg or not msg.strip():
        return history, "", state
    if not state:
        return history + [{"role": "assistant", "content": "Please upload a document first."}], "", state
    context = "Summary already shown to the patient:\n" + state["summary"]
    past = [{"role": h["role"], "content": h["content"]} for h in history[-6:]]
    messages = ([{"role": "user", "content": context},
                 {"role": "assistant", "content": "Okay, I have the summary."}]
                + past + [{"role": "user", "content": msg}])
    try:
        out = state["agent"].invoke({"messages": messages})
        answer = out["messages"][-1].content
    except Exception as e:
        answer = f"Sorry, something went wrong: {e}"
    answer += f"\n\n*{DISCLAIMER}*"
    history = history + [{"role": "user", "content": msg},
                         {"role": "assistant", "content": answer}]
    return history, "", state

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="sky"),
               css=CSS, title="Patient Record Simplifier") as demo:
    gr.HTML('<div class="hero"><h1>Patient Record Simplifier</h1>'
            '<p>Upload your medical report and get a plain-language summary instantly.</p></div>')
    state = gr.State({})
    up = gr.File(label="Upload your lab report / discharge summary / prescription",
                 type="filepath", file_types=[".pdf", ".txt"])
    cards = gr.HTML()
    who = gr.Markdown()
    gr.HTML(f'<div class="disc"><b>Disclaimer:</b> {DISCLAIMER}</div>')
    chat = gr.Chatbot(type="messages", label="Ask follow-up questions", height=350)
    box = gr.Textbox(placeholder="e.g. What does my TSH mean?  (press Enter)", show_label=False)

    up.change(on_upload, [up, state], [cards, who, chat, state])
    box.submit(on_chat, [box, chat, state], [chat, box, state])

demo.launch()