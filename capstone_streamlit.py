import streamlit as st
import uuid
from datetime import datetime


# ── Page configuration ────────────────────────────────────────
st.set_page_config(
    page_title="StyleCart Support Agent",
    page_icon="🛍️",
    layout="centered"
)


# ── Cache all heavy resources (loaded ONCE, not on every rerun) ──
@st.cache_resource
def load_agent():
    """
    Load the LLM, embedder, ChromaDB, and compiled LangGraph app.
    @st.cache_resource ensures this runs only once per session.
    Without this, the model would reload on every user message.
    """
    import os
    from langchain_groq import ChatGroq
    from sentence_transformers import SentenceTransformer
    import chromadb
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    from typing import TypedDict, List

    # ── State ─────────────────────────────────────────────────
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

    # ── Config ────────────────────────────────────────────────
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your_groq_api_key_here")
    MODEL_NAME = "llama-3.3-70b-versatile"
    FAITHFULNESS_THRESHOLD = 0.7
    MAX_EVAL_RETRIES = 2
    SLIDING_WINDOW = 6

    # ── Resources ─────────────────────────────────────────────
    llm = ChatGroq(api_key=GROQ_API_KEY, model_name=MODEL_NAME)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.Client()
    try:
        client.delete_collection("stylecart_kb")
    except:
        pass
    collection = client.create_collection("stylecart_kb")

    documents = [
        {"id": "doc_001", "topic": "Return Policy", "text": "StyleCart accepts product returns within 7 days of delivery. Items must be unused, unwashed, and in their original condition with all tags attached. Items that have been worn, washed, damaged, or altered are not eligible. To initiate a return, go to My Orders and click Request Return, or message WhatsApp at +91-98765-43210. Pickup is scheduled within 2 business days after approval. Refunds are processed within 5-7 business days after the item is received. Refunds go to the original payment method. COD order refunds need bank account details."},
        {"id": "doc_002", "topic": "Shipping and Delivery", "text": "StyleCart ships to all major cities across India. Orders are dispatched within 24 hours on business days. Standard delivery takes 3 to 5 business days. Express delivery costs Rs. 99 extra and delivers in 2 business days. Remote areas may take 7 to 10 business days. Free standard shipping on orders above Rs. 999. Orders below Rs. 999 have a flat fee of Rs. 49. A tracking link is sent via SMS and email after dispatch. International shipping is not available."},
        {"id": "doc_003", "topic": "Payment Methods", "text": "StyleCart accepts UPI (Google Pay, PhonePe, Paytm), credit cards (Visa, Mastercard, Rupay), debit cards, and net banking from all major Indian banks. EMI is available on credit cards above Rs. 3,000 for 3, 6, and 12 months. Cash on Delivery is available for orders up to Rs. 5,000 at serviceable pin codes. Prepaid orders receive a 5% instant discount. COD orders have Rs. 50 handling fee. Payments are secure and PCI-compliant."},
        {"id": "doc_004", "topic": "Order Tracking", "text": "Track orders two ways: a tracking link is sent via SMS and email after dispatch, or log in to My Orders and click the order for live status. Statuses: Order Placed, Payment Confirmed, Processing, Dispatched, Out for Delivery, Delivered. If Dispatched but no movement for 3 days, contact support with your Order ID."},
        {"id": "doc_005", "topic": "Exchange Process", "text": "Exchanges allowed within 7 days of delivery, subject to stock availability. Go to My Orders and click Request Exchange. Customers pay return shipping. StyleCart ships replacement free. Exchanges not available on sale or discounted items. If desired size/colour out of stock, a full refund is offered."},
        {"id": "doc_006", "topic": "Order Cancellation Policy", "text": "Orders can be cancelled within 2 hours of placing, if not yet dispatched. Go to My Orders and click Cancel Order. If in Processing or Dispatched status, cancellation is not possible — wait for delivery and initiate a return. Prepaid cancellations refunded in 5-7 business days. COD cancellations need no refund. Repeated cancellations may suspend COD access."},
        {"id": "doc_007", "topic": "Size Guide", "text": "StyleCart offers sizes XS, S, M, L, XL, XXL, 3XL for tops, kurtas, and dresses. Bottoms: waist 26 to 40 inches. Chest measurements: XS=32, S=34, M=36, L=38, XL=40, XXL=42, 3XL=44 inches. When between sizes, size up for comfort. Sizes may vary between categories."},
        {"id": "doc_008", "topic": "Loyalty Points Program", "text": "StyleCoins: earn 1 coin per Rs. 10 on prepaid orders. COD orders earn coins after successful delivery. 100 StyleCoins = Rs. 10 discount. Max 500 StyleCoins (Rs. 50) redeemable per order. Expire 12 months after earning. Check balance in My Rewards. Reversed on returned or cancelled orders."},
        {"id": "doc_009", "topic": "COD and Prepaid Offers", "text": "COD for orders up to Rs. 5,000 at serviceable pin codes, with Rs. 50 handling fee. Prepaid orders (UPI, card, net banking) get 5% instant discount. Prepaid dispatched faster. First-time customers get 10% off with code WELCOME10 (one-time use, prepaid only). EMI only on prepaid credit card payments above Rs. 3,000."},
        {"id": "doc_010", "topic": "Customer Support and Escalation", "text": "Support: Monday to Saturday, 9 AM to 7 PM IST. WhatsApp: +91-98765-43210, email: support@stylecart.in, or live chat on website/app. Response within 4 hours during business hours. Unresolved after 48 hours: escalate to grievance@stylecart.in, responded within 24 hours. Have Order ID ready. No phone call support."},
    ]

    texts = [d["text"] for d in documents]
    embeddings = embedder.encode(texts).tolist()
    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=[d["id"] for d in documents],
        metadatas=[{"topic": d["topic"]} for d in documents]
    )

    # ── Node functions ────────────────────────────────────────
    def memory_node(state):
        messages = state.get("messages", [])
        question = state["question"]
        messages = messages + [{"role": "user", "content": question}]
        messages = messages[-SLIDING_WINDOW:]
        customer_name = state.get("customer_name", "")
        if "my name is" in question.lower():
            parts = question.lower().split("my name is")
            if len(parts) > 1:
                customer_name = parts[1].strip().split()[0].capitalize()
        return {"messages": messages, "customer_name": customer_name, "eval_retries": 0}

    def router_node(state):
        question = state["question"]
        messages = state.get("messages", [])
        history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages[-4:]])
        prompt = f"""You are a router for StyleCart customer support.
Reply with ONE WORD only: retrieve, tool, or memory_only.
- retrieve: questions about store policies, shipping, returns, payments, sizes, exchanges, loyalty points
- tool: questions that require today's current date
- memory_only: greetings, thanks, or when answer is in conversation history

History: {history_text}
Question: {question}
Route:"""
        response = llm.invoke(prompt)
        route = response.content.strip().lower()
        if route not in ["retrieve", "tool", "memory_only"]:
            route = "retrieve"
        return {"route": route}

    def retrieval_node(state):
        question = state["question"]
        query_embedding = embedder.encode([question]).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=3)
        chunks = results["documents"][0]
        metadatas = results["metadatas"][0]
        context_parts = []
        sources = []
        for chunk, meta in zip(chunks, metadatas):
            topic = meta["topic"]
            context_parts.append(f"[{topic}]\n{chunk}")
            sources.append(topic)
        return {"retrieved": "\n\n".join(context_parts), "sources": sources}

    def skip_retrieval_node(state):
        return {"retrieved": "", "sources": []}

    def tool_node(state):
        try:
            now = datetime.now()
            tool_result = f"Today is {now.strftime('%A, %d %B %Y')}. Time: {now.strftime('%I:%M %p')} IST."
        except Exception as e:
            tool_result = f"Could not fetch date: {str(e)}"
        return {"tool_result": tool_result, "retrieved": "", "sources": []}

    def answer_node(state):
        question = state["question"]
        retrieved = state.get("retrieved", "")
        tool_result = state.get("tool_result", "")
        messages = state.get("messages", [])
        customer_name = state.get("customer_name", "")
        eval_retries = state.get("eval_retries", 0)

        history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages[:-1]])
        name_instruction = f"Address the customer as {customer_name}." if customer_name else ""
        retry_instruction = "Be extra careful to use ONLY the context below — no outside knowledge." if eval_retries >= 1 else ""

        system_prompt = f"""You are a helpful customer support assistant for StyleCart, an Indian fashion e-commerce store.
STRICT RULE: Answer ONLY from the CONTEXT provided. Do not use general knowledge.
If the answer is not in the context, say: "I don't have that specific information. Please contact our team on WhatsApp at +91-98765-43210 or email support@stylecart.in."
{name_instruction} {retry_instruction}"""

        context_block = ""
        if retrieved:
            context_block += f"\n\nKNOWLEDGE BASE:\n{retrieved}"
        if tool_result:
            context_block += f"\n\nTOOL RESULT:\n{tool_result}"

        user_msg = f"Conversation history:\n{history_text}\n{context_block}\n\nCustomer: {question}\nAnswer:"
        response = llm.invoke(f"{system_prompt}\n\n{user_msg}")
        return {"answer": response.content.strip()}

    def eval_node(state):
        answer = state.get("answer", "")
        retrieved = state.get("retrieved", "")
        eval_retries = state.get("eval_retries", 0)
        if not retrieved:
            return {"faithfulness": 1.0, "eval_retries": eval_retries}
        prompt = f"""Score faithfulness 0.0 to 1.0. Does the ANSWER use only information from CONTEXT?
CONTEXT: {retrieved}
ANSWER: {answer}
Reply with a number only:"""
        try:
            response = llm.invoke(prompt)
            faith = float(response.content.strip())
            faith = max(0.0, min(1.0, faith))
        except:
            faith = 0.5
        return {"faithfulness": faith, "eval_retries": eval_retries + 1}

    def save_node(state):
        messages = state.get("messages", [])
        answer = state.get("answer", "")
        return {"messages": messages + [{"role": "assistant", "content": answer}]}

    def route_decision(state):
        r = state.get("route", "retrieve")
        return "tool" if r == "tool" else ("skip" if r == "memory_only" else "retrieve")

    def eval_decision(state):
        faith = state.get("faithfulness", 1.0)
        retries = state.get("eval_retries", 0)
        if faith >= FAITHFULNESS_THRESHOLD or retries >= MAX_EVAL_RETRIES:
            return "save"
        return "answer"

    # ── Graph ─────────────────────────────────────────────────
    g = StateGraph(CapstoneState)
    g.add_node("memory", memory_node)
    g.add_node("router", router_node)
    g.add_node("retrieve", retrieval_node)
    g.add_node("skip", skip_retrieval_node)
    g.add_node("tool", tool_node)
    g.add_node("answer", answer_node)
    g.add_node("eval", eval_node)
    g.add_node("save", save_node)
    g.set_entry_point("memory")
    g.add_edge("memory", "router")
    g.add_edge("retrieve", "answer")
    g.add_edge("skip", "answer")
    g.add_edge("tool", "answer")
    g.add_edge("answer", "eval")
    g.add_edge("save", END)
    g.add_conditional_edges("router", route_decision, {"retrieve": "retrieve", "skip": "skip", "tool": "tool"})
    g.add_conditional_edges("eval", eval_decision, {"answer": "answer", "save": "save"})

    compiled_app = g.compile(checkpointer=MemorySaver())
    return compiled_app


# ── Load the agent (cached) ───────────────────────────────────
compiled_app = load_agent()


# ── Session state initialisation ──────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []   # Chat display messages

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())   # Unique per session


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title("🛍️ StyleCart")
    st.markdown("**AI Customer Support Agent**")
    st.markdown("---")
    st.markdown("**Topics I can help with:**")
    topics = [
        "📦 Return Policy",
        "🚚 Shipping & Delivery",
        "💳 Payment Methods",
        "🔄 Order Exchanges",
        "❌ Order Cancellation",
        "📏 Size Guide",
        "⭐ StyleCoins Loyalty",
        "💰 COD & Prepaid Offers",
        "📍 Order Tracking",
        "🆘 Customer Support",
    ]
    for t in topics:
        st.markdown(f"- {t}")

    st.markdown("---")
    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.markdown("---")
    st.caption(f"Session ID: `{st.session_state.thread_id[:8]}...`")


# ── Main chat UI ──────────────────────────────────────────────
st.title("🛍️ StyleCart Support")
st.caption("Ask me anything about orders, returns, shipping, payments, and more!")

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Welcome message on first load
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("👋 Hello! I'm the StyleCart support assistant. How can I help you today? Feel free to ask about returns, shipping, payments, sizes, or anything else!")

# Chat input
if prompt := st.chat_input("Type your question here..."):

    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Run the agent
    with st.chat_message("assistant"):
        with st.spinner("Looking that up for you..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = compiled_app.invoke({"question": prompt}, config=config)
            answer = result.get("answer", "I'm sorry, I couldn't generate a response. Please try again.")

        st.markdown(answer)

        # Show source topics in an expander
        sources = result.get("sources", [])
        if sources:
            with st.expander("📚 Sources consulted", expanded=False):
                for s in sources:
                    st.markdown(f"- {s}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
