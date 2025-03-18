from typing import Callable, List, Tuple, Any
from langgraph.graph import StateGraph, END
from azure_search_manager import AzureSearchManager
from langchain.chat_models.base import BaseChatModel

from azure_search_rag.nodes import ChatState, CONVERSATION_GRAPH
from functools import partial

# No need to import BaseModel and Field again as they are already imported from pydantic above

# Initialize the vector store and embeddings
class RAGChain:
    search_manager: AzureSearchManager
    chat_model: BaseChatModel
    system_prompt : Callable[[List[str]], str]

    def __init__(self, search_manager: AzureSearchManager = None, chat_model: BaseChatModel = None, system_prompt: Callable[[List[str]], str] = None):
        self.search_manager = search_manager
        # Configure LLM for structured output using the cited_answer model
        self.chat_model = chat_model#, include_json_schema=True)
        self.workflow = None
        self.debug_mode=False
        
        # Set system prompt first
        if system_prompt is None:
            from prompts import system_prompt_template
            self.system_prompt = system_prompt_template
        else:
            self.system_prompt = system_prompt
            
        # Initialize the graph after system_prompt is set
        self.graph = self.build_rag_chat_graph()
        # self.graph.get_graph().draw_mermaid_png(output_file_path="graph.png")

    def build_rag_chat_graph(self) -> StateGraph:
        """Create the LangGraph for our RAG chatbot."""
        self.workflow = StateGraph(ChatState)
        
        # Add nodes
        self.workflow.add_node("root", CONVERSATION_GRAPH["root"])
        self.workflow.add_node("retrieve_documents", partial(CONVERSATION_GRAPH["retrieve_documents"], search_manager=self.search_manager))
        self.workflow.add_node("generate_context", partial(CONVERSATION_GRAPH["generate_context"], system_prompt=self.system_prompt))
        self.workflow.add_node("generate_response", partial(CONVERSATION_GRAPH["generate_response"], chat_model=self.chat_model))

        # Set the entry point
        self.workflow.set_entry_point("root")
        
        # Add conditional edges
        self.workflow.add_edge("root", "retrieve_documents")
        # self.workflow.add_conditional_edges(
        #     "root",
        #     self.should_retrieve,
        #     {
        #         "retrieve": "retrieve_documents",
        #         "no_retrieve": "generate_context"
        #     }
        # )
        
        # Add other edges
        self.workflow.add_edge("retrieve_documents", "generate_context")
        self.workflow.add_edge("generate_context", "generate_response")
        self.workflow.add_edge("generate_response", END)
        
        # Compile the graph
        return self.workflow.compile()

    async def process_query(self, query: str, chat_history: List[Tuple[str, str]] = None) -> Tuple[str, List[Tuple[str, str]]]:
        """Process a query and return the response and updated chat history."""
        if chat_history is None:
            chat_history = []
        
        # Prepare the initial state
        initial_state = ChatState(
            messages=[{"role": "user", "content": query}],
            chat_history=chat_history,
            current_query=query,
            retrieved_documents=[],
            debug_mode=self.debug_mode
        )
        
        # Run the graph asynchronously
        final_state = await self.graph.ainvoke(initial_state)
        
        # Extract the latest AI response
        latest_response = final_state['messages'][-1]['content']
        
        return latest_response, final_state['chat_history']
    
    async def run(self):
        # Simple chat loop
        chat_history = []
        print("RAG Chatbot initialized. Type 'exit' to quit.")
        
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == "exit":
                break
            
            response, chat_history = await self.process_query(user_input, chat_history)
            print(f"\nChatbot: {response}")