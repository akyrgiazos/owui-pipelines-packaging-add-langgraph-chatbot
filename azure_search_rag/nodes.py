from typing import Callable, Dict, List, Tuple, Any
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.chat_models.base import BaseChatModel

class ChatState(BaseModel):
    """Represents the state of our chat system."""
    messages: List[Dict[str, str]] = Field(default_factory=list)
    chat_history: List[Tuple[str, str]] = Field(default_factory=list)
    current_query: str = ""
    retrieved_documents: List[Dict[str,Any]] = Field(default_factory=list)
    debug_mode: bool = False
    should_retrieve: str = "retrieve"

async def retrieve_documents(state: ChatState, search_manager) -> ChatState:
    """Retrieve relevant documents from the vector store based on the query."""
    # Retrieve documents
    retrieved_docs = search_manager.search(state.current_query)

    # Update state with retrieved documents
    return ChatState(
        messages=state.messages,
        chat_history=state.chat_history,
        current_query=state.current_query,
        retrieved_documents=retrieved_docs
    )
    
def format_content(state: ChatState) -> str:
    retrieved_documents = state.retrieved_documents[:]
    retrieved_documents.sort(key=lambda x: x['chunk_id'], reverse=False)
    doc_context = "\n\n".join([
        f"Document {doc['title']}:\n{doc['content']}"
        for i, doc in enumerate(state.retrieved_documents)
    ])
    return doc_context
    
async def generate_context(state: ChatState, system_prompt: Callable[[List[str]], str]) -> ChatState:
    """Generate a context string from retrieved documents and chat history."""
    # Format retrieved documents
    doc_context = format_content(state)
    
    # Format the most recent messages (last 5 exchanges)
    recent_chat_history = state.chat_history[-5:] if state.chat_history else []
    chat_context = "\n".join([
        f"Human: {human}\nAI: {ai}"
        for human, ai in recent_chat_history
    ])
    
    # Create the context message
    context_message = {
        "role": "system",
        "content": system_prompt(chat_context=chat_context, doc_context=doc_context)
    }

    
    # Update the messages with the context
    updated_messages = [context_message] + state.messages
    
    return ChatState(
        messages=updated_messages,
        chat_history=state.chat_history,
        current_query=state.current_query,
        retrieved_documents=state.retrieved_documents
        )
    
async def generate_response(state: ChatState, chat_model: BaseChatModel) -> ChatState:
    """Generate a response using the LLM based on the context and query."""
    try:
        # Format system message to explicitly request JSON output
        system_message = state.messages[0]  # First message is the system message from generate_context
        system_content = (
            system_message["content"]
        )
        
        # Create message format that the model expects with explicit JSON instruction
        formatted_messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=state.current_query)
        ]
        
        # Generate the response with structured output and validate
        response = await chat_model.ainvoke(formatted_messages)

        response_text = response.content
        
        # Update chat history
        updated_chat_history = state.chat_history + [(state.current_query, response_text)]
        
        # Update messages
        updated_messages = state.messages + [
            {"role": "assistant", "content": response_text}
        ]
        
        return ChatState(
            messages=updated_messages,
            chat_history=updated_chat_history,
            current_query=state.current_query,
            retrieved_documents=state.retrieved_documents,
            debug_mode=state.debug_mode,
            should_retrieve=state.should_retrieve
        )
        
    except Exception as e:
        # Log the error if needed
        print(f"Error in generate_response: {str(e)}")

async def route_query(state: ChatState) -> ChatState:
    """Route the query to either retrieve documents or proceed without retrieval."""
    # Keep the same routing logic but update the state properly
    should_retrieve = (
        "?" in state.current_query or
        any(keyword in state.current_query.lower() for keyword in 
            ["what", "how", "why", "when", "where", "who", "tell me about", "explain", "information"])
    )
    return ChatState(
        messages=state.messages,
        chat_history=state.chat_history,
        current_query=state.current_query,
        retrieved_documents=state.retrieved_documents,
        debug_mode=state.debug_mode,
        should_retrieve="retrieve" if should_retrieve else "no_retrieve"
    )

async def should_retrieve(state: ChatState) -> str:
    """Determine if we should retrieve documents based on the state."""
    return state.should_retrieve

CONVERSATION_GRAPH = {
    "root": route_query,
    "retrieve_documents": retrieve_documents,
    "generate_context": generate_context,
    "generate_response": generate_response
}