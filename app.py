"""
Agentic RAG application for AI blog search using LangGraph.

This module implements a retrieval-augmented generation (RAG) system with
agentic decision-making capabilities to search and answer questions about
AI-related blog posts.
"""

import logging
from typing import Annotated, Literal, Sequence
from functools import partial
from uuid import uuid4

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
import streamlit as st

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="AI Blog Search", page_icon=":mag_right:")
st.header(":blue[Agentic RAG with LangGraph:] :green[AI Blog Search]")


# Initialize session state variables if they don't exist
if 'qdrant_host' not in st.session_state:
    st.session_state.qdrant_host = ""
if 'qdrant_api_key' not in st.session_state:
    st.session_state.qdrant_api_key = ""
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ""


class GradeScore(BaseModel):
    """Binary relevance score for document grading."""

    binary_score: str = Field(description="Relevance score 'yes' or 'no'")


class AgentState(dict):
    """State dictionary for the agent workflow."""

    messages: Annotated[Sequence[BaseMessage], add_messages]


def set_sidebar() -> None:
    """Setup sidebar for API keys and configuration."""
    with st.sidebar:
        st.subheader("API Configuration")

        qdrant_host = st.text_input("Enter your Qdrant Host URL:", type="password")
        qdrant_api_key = st.text_input("Enter your Qdrant API key:", type="password")
        gemini_api_key = st.text_input("Enter your Gemini API key:", type="password")

        if st.button("Done"):
            if qdrant_host and qdrant_api_key and gemini_api_key:
                st.session_state.qdrant_host = qdrant_host
                st.session_state.qdrant_api_key = qdrant_api_key
                st.session_state.gemini_api_key = gemini_api_key
                st.success("API keys saved!")
            else:
                st.warning("Please fill all API fields")


def initialize_components() -> tuple:
    """
    Initialize components that require API keys.

    Returns:
        tuple: (embedding_model, client, db) or (None, None, None) if initialization fails.
    """
    if not all([st.session_state.qdrant_host,
                st.session_state.qdrant_api_key,
                st.session_state.gemini_api_key]):
        return None, None, None

    try:
        # Initialize embedding model with API key
        embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=st.session_state.gemini_api_key
        )

        # Initialize Qdrant client
        client = QdrantClient(
            st.session_state.qdrant_host,
            api_key=st.session_state.qdrant_api_key
        )

        # Initialize vector store
        db = QdrantVectorStore(
            client=client,
            collection_name="qdrant_db",
            embedding=embedding_model
        )

        return embedding_model, client, db

    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
        logger.exception("Failed to initialize components")
        return None, None, None


def grade_documents(state: AgentState) -> Literal["generate", "rewrite"]:
    """
    Determine whether retrieved documents are relevant to the question.

    Args:
        state: The current agent state containing messages

    Returns:
        str: "generate" if documents are relevant, "rewrite" otherwise
    """
    logger.info("---CHECK RELEVANCE---")

    # LLM for grading
    model = ChatGoogleGenerativeAI(
        api_key=st.session_state.gemini_api_key,
        temperature=0,
        model="gemini-2.0-flash",
        streaming=True
    )

    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(GradeScore)

    # Prompt for relevance grading
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question.
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Create chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        logger.info("---DECISION: DOCS RELEVANT---")
        return "generate"
    else:
        logger.info("---DECISION: DOCS NOT RELEVANT---")
        return "rewrite"


def agent(state: AgentState, tools: list) -> dict:
    """
    Invoke the agent model to generate a response.

    Args:
        state: The current agent state
        tools: List of tools available to the agent

    Returns:
        dict: Updated state with agent response appended
    """
    logger.info("---CALL AGENT---")
    messages = state["messages"]
    model = ChatGoogleGenerativeAI(
        api_key=st.session_state.gemini_api_key,
        temperature=0,
        streaming=True,
        model="gemini-2.0-flash"
    )
    model = model.bind_tools(tools)
    response = model.invoke(messages)

    return {"messages": [response]}


def rewrite(state: AgentState) -> dict:
    """
    Transform the query to produce a better question.

    Args:
        state: The current agent state

    Returns:
        dict: Updated state with rephrased question
    """
    logger.info("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f""" \n
                    Look at the input and try to reason about the underlying semantic intent / meaning. \n
                    Here is the initial question:
                    \n ------- \n
                    {question}
                    \n ------- \n
                    Formulate an improved question: """,
        )
    ]

    # Grader model
    model = ChatGoogleGenerativeAI(
        api_key=st.session_state.gemini_api_key,
        temperature=0,
        model="gemini-2.0-flash",
        streaming=True
    )
    response = model.invoke(msg)
    return {"messages": [response]}


def generate(state: AgentState) -> dict:
    """
    Generate answer based on retrieved documents.

    Args:
        state: The current agent state

    Returns:
        dict: Updated state with generated answer
    """
    logger.info("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    # Initialize a Chat Prompt Template
    prompt_template = hub.pull("rlm/rag-prompt")

    # Initialize a Generator (i.e. Chat Model)
    chat_model = ChatGoogleGenerativeAI(
        api_key=st.session_state.gemini_api_key,
        model="gemini-2.0-flash",
        temperature=0,
        streaming=True
    )

    # Initialize a Output Parser
    output_parser = StrOutputParser()

    # RAG Chain
    rag_chain = prompt_template | chat_model | output_parser

    response = rag_chain.invoke({"context": docs, "question": question})

    return {"messages": [response]}


def get_graph(retriever_tool):
    """
    Create and compile the LangGraph workflow.

    Args:
        retriever_tool: The retriever tool for the graph

    Returns:
        CompiledGraph: The compiled workflow graph
    """
    tools = [retriever_tool]

    # Define a new graph
    workflow = StateGraph(AgentState)

    # Use partial to pass tools to the agent function
    workflow.add_node("agent", partial(agent, tools=tools))

    # Add other nodes
    retrieve = ToolNode(tools)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("rewrite", rewrite)
    workflow.add_node("generate", generate)

    # Add edges
    workflow.add_edge(START, "agent")

    # Conditional edges for tool calling decision
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "retrieve",
            END: END,
        },
    )

    # Conditional edges for document relevance
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
    )
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", "agent")

    # Compile
    graph = workflow.compile()

    return graph


def generate_message(graph, inputs: dict) -> str:
    """
    Generate a response message using the compiled graph.

    Args:
        graph: The compiled LangGraph workflow
        inputs: Input dictionary with messages

    Returns:
        str: Generated message from the graph
    """
    generated_message = ""

    for output in graph.stream(inputs):
        for key, value in output.items():
            if key == "generate" and isinstance(value, dict):
                generated_message = value.get("messages", [""])[0]

    return generated_message


def add_documents_to_qdrant(url: str, db: QdrantVectorStore) -> bool:
    """
    Load and add documents from a URL to the Qdrant vector database.

    Args:
        url: URL of the blog post to load
        db: Qdrant vector store instance

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        docs = WebBaseLoader(url).load()
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=100, chunk_overlap=50
        )
        doc_chunks = text_splitter.split_documents(docs)
        uuids = [str(uuid4()) for _ in range(len(doc_chunks))]
        db.add_documents(documents=doc_chunks, ids=uuids)
        logger.info(f"Successfully added {len(doc_chunks)} documents from {url}")
        return True
    except Exception as e:
        st.error(f"Error adding documents: {str(e)}")
        logger.exception(f"Error adding documents from {url}")
        return False


def main() -> None:
    """Main application entry point."""
    set_sidebar()

    # Check if API keys are set
    if not all([st.session_state.qdrant_host,
                st.session_state.qdrant_api_key,
                st.session_state.gemini_api_key]):
        st.warning("Please configure your API keys in the sidebar first")
        return

    # Initialize components
    embedding_model, client, db = initialize_components()
    if not all([embedding_model, client, db]):
        return

    # Initialize retriever and tools
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_blog_posts",
        "Search and return information about blog posts on LLMs, LLM agents, prompt engineering, and adversarial attacks on LLMs.",
    )

    # URL input section
    url = st.text_input(
        ":link: Paste the blog link:",
        placeholder="e.g., https://lilianweng.github.io/posts/2023-06-23-agent/"
    )
    if st.button("Enter URL"):
        if url:
            with st.spinner("Processing documents..."):
                if add_documents_to_qdrant(url, db):
                    st.success("Documents added successfully!")
                else:
                    st.error("Failed to add documents")
        else:
            st.warning("Please enter a URL")

    # Query section
    graph = get_graph(retriever_tool)
    query = st.text_area(
        ":bulb: Enter your query about the blog post:",
        placeholder="e.g., What does Lilian Weng say about the types of agent memory?"
    )

    if st.button("Submit Query"):
        if not query:
            st.warning("Please enter a query")
            return

        inputs = {"messages": [HumanMessage(content=query)]}
        with st.spinner("Generating response..."):
            try:
                response = generate_message(graph, inputs)
                st.write(response)
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                logger.exception("Error during response generation")

    st.markdown("---")
    st.markdown(
        "Powered by [LangChain](https://www.langchain.com/) | "
        "[LangGraph](https://langchain-ai.github.io/langgraph/) | "
        "[Qdrant](https://qdrant.tech/) by [Rishi Chhabra](https://github.com/rchhabra13)"
    )


if __name__ == "__main__":
    main()
