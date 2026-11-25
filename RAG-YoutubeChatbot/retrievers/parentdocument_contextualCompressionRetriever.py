
import os
from dotenv import load_dotenv

# Document abstraction
from langchain_core.documents import Document

# Text splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LLM and embedding models
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Vector database
from langchain_chroma import Chroma

# Parent document storage
from langchain_classic.storage import InMemoryStore

# Retrievers
from langchain_classic.retrievers import (
    ParentDocumentRetriever,
    ContextualCompressionRetriever
)

# Document compressors and transformers
from langchain_classic.retrievers.document_compressors import (
    EmbeddingsFilter,
    LLMChainExtractor,
    DocumentCompressorPipeline
)


# --------------------------------------------------
# 1. ENVIRONMENT CONFIGURATION
# --------------------------------------------------

load_dotenv()

# Embedding model converts text into numerical vectors.
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# LLM used by the contextual compression stage
# to extract query-relevant information.
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# --------------------------------------------------
# 2. DOCUMENT LOADING
# --------------------------------------------------

# Sample document for demonstration.
# Replace this with documents loaded from your PDFs,
# websites, or other data sources.

documents = [
    Document(
        page_content="""
        Gradient descent is an optimization algorithm used
        in machine learning to minimize a loss function.

        It works by calculating the gradient of the loss
        function with respect to model parameters.

        The weights are updated in the opposite direction
        of the gradient.

        The learning rate controls the size of each update.

        Gradient descent repeats this process until
        convergence or another stopping condition is reached.

        Machine learning has several categories, including
        supervised learning, unsupervised learning and
        reinforcement learning.

        Supervised learning uses labelled training data.
        """,
        metadata={"source": "machine_learning.txt"}
    )
]


# --------------------------------------------------
# 3. PARENT AND CHILD CHUNKING
# --------------------------------------------------

# Parent chunks preserve broader contextual information.
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

# Smaller child chunks provide finer-grained retrieval.
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)


# --------------------------------------------------
# 4. VECTOR STORE AND DOCUMENT STORE
# --------------------------------------------------

# Chroma stores child chunks and their embeddings.
vectorstore = Chroma(
    collection_name="parent_child_collection",
    embedding_function=embeddings
)

# InMemoryStore stores the larger parent documents.
# Child chunks reference their corresponding parents
# through parent IDs in their metadata.
docstore = InMemoryStore()


# --------------------------------------------------
# 5. PARENT DOCUMENT RETRIEVER
# --------------------------------------------------

parent_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    parent_splitter=parent_splitter,
    child_splitter=child_splitter,
    search_kwargs={"k": 3}
)

# Adds documents and internally:
# 1. Creates parent chunks.
# 2. Creates child chunks from each parent.
# 3. Embeds and stores child chunks in Chroma.
# 4. Stores parent documents in the Docstore.

parent_retriever.add_documents(documents)


# --------------------------------------------------
# 6. DOCUMENT COMPRESSION PIPELINE
# --------------------------------------------------

# STAGE 1: EMBEDDINGS FILTER
#
# Compares retrieved parent documents with the query
# using embedding similarity.
#
# Documents below the similarity threshold are removed.
# This is document-level filtering, not sentence-level
# extraction.

embeddings_filter = EmbeddingsFilter(
    embeddings=embeddings,
    similarity_threshold=0.65
)


# STAGE 2: LLM CHAIN EXTRACTOR
#
# Examines the remaining documents and the user query.
# Extracts only the relevant passages from each document.
#
# Unlike EmbeddingsFilter, this can remove irrelevant
# information within an otherwise relevant parent document.

llm_compressor = LLMChainExtractor.from_llm(
    llm
)


# Combine both stages into one sequential pipeline.
#
# First: EmbeddingsFilter
# Second: LLMChainExtractor

compressor_pipeline = DocumentCompressorPipeline(
    transformers=[
        embeddings_filter,
        llm_compressor
    ]
)


# --------------------------------------------------
# 7. CONTEXTUAL COMPRESSION RETRIEVER
# --------------------------------------------------

# Wrap ParentDocumentRetriever with the compression pipeline.
#
# Retrieval flow:
# 1. ParentDocumentRetriever searches child embeddings.
# 2. Retrieves corresponding parent documents.
# 3. EmbeddingsFilter removes irrelevant parent documents.
# 4. LLMChainExtractor extracts relevant passages.
# 5. Returns compressed documents.

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor_pipeline,
    base_retriever=parent_retriever
)


# --------------------------------------------------
# 8. TEST RETRIEVAL
# --------------------------------------------------

query = "How does gradient descent update weights?"


# First, inspect the original parent documents.
parent_docs = parent_retriever.invoke(query)

print("\n========== PARENT DOCUMENTS ==========")

for i, doc in enumerate(parent_docs, start=1):
    print(f"\nParent Document {i}:")
    print(doc.page_content)


# Next, retrieve compressed documents.
compressed_docs = compression_retriever.invoke(query)

print("\n========== COMPRESSED DOCUMENTS ==========")

for i, doc in enumerate(compressed_docs, start=1):
    print(f"\nCompressed Document {i}:")
    print(doc.page_content)
    print("Metadata:", doc.metadata)


# --------------------------------------------------
# 9. OPTIONAL: PASS COMPRESSED CONTEXT TO FINAL LLM
# --------------------------------------------------

# Join the compressed documents into a context string.

context = "\n\n".join(
    doc.page_content for doc in compressed_docs
)

prompt = f"""
Answer the question using only the following context.

Context:
{context}

Question:
{query}

If the context does not contain enough information,
say that you do not have enough information.
"""

response = llm.invoke(prompt)

print("\n========== FINAL ANSWER ==========")
print(response.content)
