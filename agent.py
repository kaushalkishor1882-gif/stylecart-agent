# ============================================================
# agent.py — StyleCart Agentic AI Capstone
# ============================================================


import os
from datetime import datetime
from typing import TypedDict, List

from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
import chromadb
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


# ── Configuration ────────────────────────────────────────────
GROQ_API_KEY = "PASTE_YOUR_REAL_KEY_HERE"
MODEL_NAME = "llama-3.3-70b-versatile"
FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES = 2
SLIDING_WINDOW = 6


# ── State ─────────────────────────────────────────────────────
class CapstoneState(TypedDict):
    question: str
    messages: List[dict]
    route: str
    retrieved: str
    sources: List[str]
    tool_result: str
    answer: str
    faithfulness: float
    eval_retries: int
    customer_name: str


# ── Resource initialisation ───────────────────────────────────
print("[agent.py] Loading resources...")
llm = ChatGroq(api_key=GROQ_API_KEY, model_name=MODEL_NAME)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client()
try:
    client.delete_collection("stylecart_kb")
except:
    pass
collection = client.create_collection("stylecart_kb")

_documents = [
    {"id": "doc_001", "topic": "Return Policy", "text": "StyleCart accepts product returns within 7 days of delivery. Items must be unused, unwashed, and in their original condition with all tags attached. Items that have been worn, washed, damaged, or altered are not eligible. To initiate a return, go to My Orders and click Request Return, or message WhatsApp at +91-98765-43210. Pickup is scheduled within 2 business days after approval. Refunds are processed within 5-7 business days after the item is received. Refunds go to the original payment method. COD order refunds need bank account details."},
    {"id": "doc_002", "topic": "Shipping and Delivery", "text": "StyleCart ships to all major cities across India. Orders are dispatched within 24 hours on business days. Standard delivery takes 3 to 5 business days. Express delivery costs Rs. 99 extra and delivers in 2 business days. Remote areas may take 7 to 10 business days. Free standard shipping on orders above Rs. 999. Orders below Rs. 999 have a flat fee of Rs. 49. A tracking link is sent via SMS and email after dispatch. International shipping is not available."},
    {"id": "doc_003", "topic": "Payment Methods", "text": "StyleCart accepts UPI (Google Pay, PhonePe, Paytm), credit cards (Visa, Mastercard, Rupay), debit cards, and net banking from all major Indian banks. EMI is available on credit cards above Rs. 3,000 for 3, 6, and 12 months. Cash on Delivery is available for orders up to Rs. 5,000 at serviceable pin codes. Prepaid orders receive a 5% instant discount. COD orders have Rs. 50 handling fee. Payments are secure and PCI-compliant."},
    {"id": "doc_004", "topic": "Order Tracking", "text": "Track orders two ways: a tracking link is sent via SMS and email after dispatch, or log in to My Orders and click the order for live status. Statuses: Order Placed, Payment Confirmed, Processing, Dispatched, Out for Delivery, Delivered. If Dispatched but no movement for 3 days, contact support with your Order ID."},
    {"id": "doc_005", "topic": "Exchange Process", "text": "Exchanges allowed within 7 days of delivery, subject to stock availability. Go to My Orders and click Request Exchange. Customers pay return shipping. StyleCart ships replacement free. Exchanges not available on sale or discounted items. If desired size/colour out of stock, a full refund is offered."},
    {"id": "doc_006", "topic": "Order Cancellation Policy", "text": "Orders can be cancelled within 2 hours of placing, if not yet dispatched. Go to My Orders and click Cancel Order. If in Processing or Dispatched status, cancellation is not possible. Prepaid cancellations refunded in 5-7 business days. COD cancellations need no refund. Repeated cancellations may suspend COD access."},
    {"id": "doc_007", "topic": "Size Guide", "text": "StyleCart offers sizes XS, S, M, L, XL, XXL, 3XL for tops, kurtas, dresses. Bottoms: waist 26 to 40 inches. Chest measurements: XS=32, S=34, M=36, L=38, XL=40, XXL=42, 3XL=44 inches. When between sizes, size up. Sizes may vary between categories."},
    {"id": "doc_008", "topic": "Loyalty Points Program", "text": "StyleCoins: earn 1 coin per Rs. 10 on prepaid orders. COD orders earn coins after successful delivery. 100 StyleCoins = Rs. 10 discount. Max 500 StyleCoins redeemable per order. Expire 12 months after earning. Check balance in My Rewards. Reversed on returned or cancelled orders."},
    {"id": "doc_009", "topic": "COD and Prepaid Offers", "text": "COD for orders up to Rs. 5,000 at serviceable pin codes, with Rs. 50 handling fee. Prepaid orders get 5% instant discount. First-time customers get 10% off with code WELCOME10 (one-time, prepaid only). EMI only on prepaid credit card payments above Rs. 3,000."},
    {"id": "doc_010", "topic": "Customer Support and Escalation", "text": "Support: Monday to Saturday, 9 AM to 7 PM IST. WhatsApp: +91-98765-43210, email: support@stylecart.in, or live chat on website/app. Response within 4 hours. Escalate unresolved issues to grievance@stylecart.in. No phone call support."},
]

_texts = [d["text"] for d in _documents]
_embeddings = embedder.encode(_texts).tolist()
collection.add(
    documents=_texts,
    embeddings=_embeddings,
    ids=[d["id"] for d in _documents],
    metadatas=[{"topic": d["topic"]} for d in _documents]
)
print(f"[agent.py] ChromaDB ready: {collection.count()} documents.")


# ── Node functions ────────────────────────────────────────────
def memory_node(state: CapstoneState) -> dict:
    messages = state.get("messages", []) + [{"role": "user", "content": state["question"]}]
    messages = messages[-SLIDING_WINDOW:]
    customer_name = state.get("customer_name", "")
    if "my name is" in state["question"].lower():
        parts = state["question"].lower().split("my name is")
        if len(parts) > 1:
            customer_name = parts[1].strip().split()[0].capitalize()
    return {"messages": messages, "customer_name": customer_name, "eval_retries": 0}


def router_node(state: CapstoneState) -> dict:
    history = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in state.get("messages", [])[-4:]])
    prompt = f"""Router for StyleCart support. Reply with ONE word: retrieve, tool, or memory_only.
- retrieve: store policies, shipping, returns, payments, sizes, loyalty, exchanges
- tool: questions needing today's date
- memory_only: greetings or answers already in history

History: {history}
Question: {state['question']}
Route:"""
    response = llm.invoke(prompt)
    route = response.content.strip().lower()
    if route not in ["retrieve", "tool", "memory_only"]:
        route = "retrieve"
    return {"route": route}


def retrieval_node(state: CapstoneState) -> dict:
    qe = embedder.encode([state["question"]]).tolist()
    results = collection.query(query_embeddings=qe, n_results=3)
    context_parts, sources = [], []
    for chunk, meta in zip(results["documents"][0], results["metadatas"][0]):
        context_parts.append(f"[{meta['topic']}]\n{chunk}")
        sources.append(meta["topic"])
    return {"retrieved": "\n\n".join(context_parts), "sources": sources}


def skip_retrieval_node(state: CapstoneState) -> dict:
    return {"retrieved": "", "sources": []}


def tool_node(state: CapstoneState) -> dict:
    try:
        now = datetime.now()
        result = f"Today is {now.strftime('%A, %d %B %Y')}. Time: {now.strftime('%I:%M %p')} IST."
    except Exception as e:
        result = f"Could not get date: {str(e)}"
    return {"tool_result": result, "retrieved": "", "sources": []}


def answer_node(state: CapstoneState) -> dict:
    history = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in state.get("messages", [])[:-1]])
    name_inst = f"Address the customer as {state.get('customer_name', '')}." if state.get("customer_name") else ""
    retry_inst = "Use ONLY the context below — no outside knowledge." if state.get("eval_retries", 0) >= 1 else ""
    sys = f"""You are StyleCart customer support. STRICT RULE: Answer ONLY from CONTEXT below. No general knowledge.
If answer not in context, say: "I don't have that info. Contact us: WhatsApp +91-98765-43210 or support@stylecart.in."
{name_inst} {retry_inst}"""
    ctx = ""
    if state.get("retrieved"):
        ctx += f"\n\nKNOWLEDGE BASE:\n{state['retrieved']}"
    if state.get("tool_result"):
        ctx += f"\n\nTOOL RESULT:\n{state['tool_result']}"
    msg = f"History:\n{history}\n{ctx}\n\nCustomer: {state['question']}\nAnswer:"
    response = llm.invoke(f"{sys}\n\n{msg}")
    return {"answer": response.content.strip()}


def eval_node(state: CapstoneState) -> dict:
    retrieved = state.get("retrieved", "")
    retries = state.get("eval_retries", 0)
    if not retrieved:
        return {"faithfulness": 1.0, "eval_retries": retries}
    prompt = f"""Score faithfulness 0.0–1.0. Does ANSWER use only CONTEXT?
CONTEXT: {retrieved}
ANSWER: {state.get('answer', '')}
Number only:"""
    try:
        faith = float(llm.invoke(prompt).content.strip())
        faith = max(0.0, min(1.0, faith))
    except:
        faith = 0.5
    return {"faithfulness": faith, "eval_retries": retries + 1}


def save_node(state: CapstoneState) -> dict:
    return {"messages": state.get("messages", []) + [{"role": "assistant", "content": state.get("answer", "")}]}


def route_decision(state: CapstoneState) -> str:
    r = state.get("route", "retrieve")
    return "tool" if r == "tool" else ("skip" if r == "memory_only" else "retrieve")


def eval_decision(state: CapstoneState) -> str:
    faith = state.get("faithfulness", 1.0)
    retries = state.get("eval_retries", 0)
    if faith >= FAITHFULNESS_THRESHOLD or retries >= MAX_EVAL_RETRIES:
        return "save"
    return "answer"


# ── Graph assembly ────────────────────────────────────────────
g = StateGraph(CapstoneState)
for name, fn in [("memory", memory_node), ("router", router_node), ("retrieve", retrieval_node),
                  ("skip", skip_retrieval_node), ("tool", tool_node), ("answer", answer_node),
                  ("eval", eval_node), ("save", save_node)]:
    g.add_node(name, fn)

g.set_entry_point("memory")
g.add_edge("memory", "router")
g.add_edge("retrieve", "answer")
g.add_edge("skip", "answer")
g.add_edge("tool", "answer")
g.add_edge("answer", "eval")
g.add_edge("save", END)
g.add_conditional_edges("router", route_decision, {"retrieve": "retrieve", "skip": "skip", "tool": "tool"})
g.add_conditional_edges("eval", eval_decision, {"answer": "answer", "save": "save"})

app = g.compile(checkpointer=MemorySaver())
print("[agent.py] Graph compiled successfully.")


# ── Public API ────────────────────────────────────────────────
def ask(question: str, thread_id: str = "default") -> dict:
    """
    Ask the StyleCart support agent a question.

    Args:
        question (str): Customer's question
        thread_id (str): Unique session ID for conversation memory

    Returns:
        dict: Contains 'answer', 'route', 'faithfulness', 'sources'
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"question": question}, config=config)
    return result
