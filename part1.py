# ============================================================
# PART 1: KNOWLEDGE BASE — StyleCart E-Commerce FAQ Agent
# ============================================================
# Run this file first to verify ChromaDB retrieval works
# before building any nodes or graph.
# ============================================================

from sentence_transformers import SentenceTransformer
import chromadb

# ── 1. Load the embedding model ──────────────────────────────
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.")

# ── 2. Define 10 knowledge base documents ────────────────────
# Each document covers ONE specific topic, 100-500 words.
# Format: {id, topic, text}

documents = [
    {
        "id": "doc_001",
        "topic": "Return Policy",
        "text": """StyleCart accepts product returns within 7 days of delivery.
To be eligible for a return, items must be unused, unwashed, and in their original
condition with all tags attached. Items that have been worn, washed, damaged, or
altered in any way are not eligible for return. To initiate a return, customers
must log in to their StyleCart account, go to 'My Orders', select the item, and
click 'Request Return'. Alternatively, customers can message the StyleCart WhatsApp
helpline at +91-98765-43210. Once the return is approved, a pickup will be scheduled
within 2 business days. Refunds are processed within 5-7 business days after the
returned item is received and inspected. Refunds are credited to the original payment
method. For COD orders, refunds are processed via bank transfer and require the
customer's bank account details."""
    },
    {
        "id": "doc_002",
        "topic": "Shipping and Delivery",
        "text": """StyleCart ships orders to all major cities and towns across India.
Orders are dispatched within 24 hours of placement on business days (Monday to
Saturday, excluding public holidays). Standard delivery takes 3 to 5 business days
from the date of dispatch. Express delivery is available for an additional charge
of Rs. 99 and delivers within 2 business days. Delivery to remote or non-serviceable
areas may take 7 to 10 business days. Free standard shipping is available on all
orders above Rs. 999. Orders below Rs. 999 are charged a flat shipping fee of Rs. 49.
A tracking link is sent via SMS and email once the order is dispatched. StyleCart
currently does not offer international shipping."""
    },
    {
        "id": "doc_003",
        "topic": "Payment Methods",
        "text": """StyleCart accepts multiple payment methods to ensure a smooth shopping
experience. Accepted payment options include UPI (Google Pay, PhonePe, Paytm),
credit cards (Visa, Mastercard, Rupay), debit cards, and net banking from all
major Indian banks. EMI options are available on credit card purchases above Rs. 3,000
for 3, 6, and 12-month tenures. Cash on Delivery (COD) is available for orders
up to Rs. 5,000 and for serviceable pin codes only. Prepaid orders (any method
other than COD) receive an instant 5% discount applied automatically at checkout.
COD orders carry an additional handling fee of Rs. 50. Payments are processed
through a secure, PCI-compliant gateway. StyleCart does not store card details."""
    },
    {
        "id": "doc_004",
        "topic": "Order Tracking",
        "text": """Customers can track their StyleCart orders in two ways. First, a
tracking link is sent automatically via SMS and email to the registered mobile
number and email address once the order is dispatched. Clicking this link opens
the courier partner's live tracking page. Second, customers can log in to their
StyleCart account, navigate to 'My Orders', and click on the specific order to
see the current status and estimated delivery date. Order statuses include:
Order Placed, Payment Confirmed, Processing, Dispatched, Out for Delivery, and
Delivered. If an order shows 'Dispatched' but no movement is seen for more than
3 days, customers should contact the StyleCart support team with the order ID."""
    },
    {
        "id": "doc_005",
        "topic": "Exchange Process",
        "text": """StyleCart allows size and colour exchanges within 7 days of delivery,
subject to stock availability. To request an exchange, customers must go to
'My Orders' in their account, select the item, and click 'Request Exchange'.
The reason for exchange and the preferred size or colour must be specified.
Once approved, customers need to ship the item back to StyleCart's warehouse.
The return shipping cost for exchanges is borne by the customer. StyleCart will
ship the replacement item free of charge once the original item is received
and quality-checked. Exchanges are not available for sale or discounted items.
If the desired size or colour is not in stock, StyleCart will offer a full
refund instead."""
    },
    {
        "id": "doc_006",
        "topic": "Order Cancellation Policy",
        "text": """Customers can cancel an order within 2 hours of placing it, provided
the order has not yet been dispatched. To cancel, go to 'My Orders' in your
StyleCart account and click 'Cancel Order'. If the order is already in Processing
or Dispatched status, it cannot be cancelled. In such cases, the customer must
wait for delivery and then initiate a return request. For prepaid orders that
are successfully cancelled within the 2-hour window, the refund is processed to
the original payment method within 5 to 7 business days. COD orders that are
cancelled do not require any refund. Repeated order cancellations by the same
customer may result in temporary suspension of COD privileges."""
    },
    {
        "id": "doc_007",
        "topic": "Size Guide",
        "text": """StyleCart offers clothing in sizes XS, S, M, L, XL, XXL, and 3XL for
tops, kurtas, and dresses. For bottoms such as trousers and jeans, waist sizes
range from 26 to 40 inches. A detailed size chart is available on every product
page. Customers are advised to measure their chest, waist, and hip before
ordering. For tops, measure the chest and refer to the chart: XS fits 32 inches,
S fits 34, M fits 36, L fits 38, XL fits 40, XXL fits 42, 3XL fits 44 inches.
When in doubt between two sizes, StyleCart recommends sizing up for comfort.
Sizes can vary slightly between product categories and brands available on
StyleCart. If a product runs small or large, this is mentioned in the product
description."""
    },
    {
        "id": "doc_008",
        "topic": "Loyalty Points Program",
        "text": """StyleCart rewards customers through its loyalty points program called
StyleCoins. Customers earn 1 StyleCoin for every Rs. 10 spent on prepaid orders.
COD orders earn StyleCoins only after the order is successfully delivered and
not returned. 100 StyleCoins are equal to a Rs. 10 discount on the next purchase.
StyleCoins can be redeemed at checkout and are applied before any coupon codes.
A maximum of 500 StyleCoins (Rs. 50 discount) can be redeemed per order.
StyleCoins expire 12 months from the date they are earned. Customers can check
their StyleCoin balance by logging into their account and visiting 'My Rewards'.
StyleCoins earned on an order are reversed if the order is returned or cancelled."""
    },
    {
        "id": "doc_009",
        "topic": "COD and Prepaid Offers",
        "text": """StyleCart offers both Cash on Delivery (COD) and prepaid payment options.
COD is available for orders up to Rs. 5,000 at serviceable pin codes only.
A handling fee of Rs. 50 is charged on all COD orders. Prepaid orders made
via UPI, credit card, debit card, or net banking automatically receive a 5%
instant discount at checkout. This discount is not available on COD orders.
Prepaid orders are also dispatched faster, as payment confirmation is instant.
First-time customers using the app or website receive an additional 10% off
on their first prepaid order using the code WELCOME10. This code is valid for
one use only and cannot be combined with other offers. EMI options are only
available on prepaid credit card payments above Rs. 3,000."""
    },
    {
        "id": "doc_010",
        "topic": "Customer Support and Escalation",
        "text": """StyleCart's customer support team is available Monday to Saturday,
9 AM to 7 PM IST. Customers can reach support through the following channels:
WhatsApp at +91-98765-43210, email at support@stylecart.in, or via the
live chat option on the StyleCart website and app. For most issues such as
returns, exchanges, and order queries, the team responds within 4 hours during
business hours. If an issue is not resolved within 48 hours, customers can
escalate to the grievance team at grievance@stylecart.in. The grievance officer
responds within 24 hours. For urgent delivery issues, always contact support
with your Order ID ready. StyleCart does not offer phone call support at this time."""
    }
]

# ── 3. Build ChromaDB in-memory collection ───────────────────
print("\nBuilding ChromaDB collection...")
client = chromadb.Client()

# Delete collection if it already exists (for re-runs)
try:
    client.delete_collection("stylecart_kb")
except:
    pass

collection = client.create_collection("stylecart_kb")

# Embed all documents
texts = [doc["text"] for doc in documents]
embeddings = embedder.encode(texts).tolist()   # .tolist() required by ChromaDB

collection.add(
    documents=texts,
    embeddings=embeddings,
    ids=[doc["id"] for doc in documents],
    metadatas=[{"topic": doc["topic"]} for doc in documents]
)

print(f"Added {collection.count()} documents to ChromaDB.")

# ── 4. RETRIEVAL TEST — verify before building graph ─────────
print("\n--- RETRIEVAL TEST ---")

test_queries = [
    "What is your return policy?",
    "How long does shipping take?",
    "Do you accept cash on delivery?",
]

for query in test_queries:
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=2
    )
    topics = [m["topic"] for m in results["metadatas"][0]]
    print(f"\nQuery : {query}")
    print(f"Retrieved topics: {topics}")

print("\n✅ Part 1 complete. Retrieval verified. Proceed to Part 2.")