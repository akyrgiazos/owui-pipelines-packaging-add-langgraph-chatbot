from typing import Callable, Dict, List, Tuple, Any
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from azure_search_rag.azure_search_manager import AzureSearchManager
from langchain.chat_models.base import BaseChatModel

# No need to import BaseModel and Field again as they are already imported from pydantic above


class cited_answer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""

    answer: str = Field(
        ...,
        description="The answer to the user question, which is based only on the given sources.",
    )
    citations: List[str] = Field(
        ...,
        description="The titles of the SPECIFIC sources which justify the answer.",
    )
# Define the state for our graph
class ChatState(BaseModel):
    """Represents the state of our chat system."""
    messages: List[Dict[str, str]] = Field(default_factory=list)
    chat_history: List[Tuple[str, str]] = Field(default_factory=list)
    current_query: str = ""
    retrieved_documents: List[Dict[str,Any]] = Field(default_factory=list)
    debug_mode: bool = False
    should_retrieve: str = "retrieve"


# Initialize the vector store and embeddings
class RAGChain:
    search_manager: AzureSearchManager
    chat_model: BaseChatModel
    system_prompt : Callable[[List[str]], str]

    def __init__(self, search_manager: AzureSearchManager = None, chat_model: BaseChatModel = None, system_prompt: Callable[List[str], str] = None):
        self.search_manager = search_manager
        # Configure LLM for structured output using the cited_answer model
        self.chat_model = chat_model#, include_json_schema=True)
        self.workflow = None
        # Initialize the graph
        self.graph = self.build_rag_chat_graph()
        # self.graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
        self.debug_mode=False
        if system_prompt is None:
            self.system_prompt = lambda chat_context, doc_context: f"""
                You are an AI assistant for a bank responsible to answer to fellow employees of the bank and NOT customers, 
                specializing in providing comprehensive information about banking products, services, and procedures. Always mention the related products.

                The "answer" field must be a string containing a detailed response that:
                - Is based only on the given sources
                - Follows the guidelines below for content organization
                - Acknowledges any limitations if information is not found in the sources

                The "citations" field must be an array of strings containing:
                - Only the titles of SPECIFIC sources that directly justify your answer
                - Must omit any sources that weren't actually used in the answer

                For the answer content, follow these formatting and structure guidelines:
                
                FORMATTING:
                - Use "**bold text**" for emphasis on important terms or section headers
                - Use "- " prefix for list items
                - Keep one blank line before and after lists
                - Use clear paragraph breaks for different sections
                

                RESPONSE STRUCTURE:
                1. For general product inquiries:
                - Begin with a high-level summary of available options
                - Group products by category
                - Include key differentiating features
                - Highlight eligibility criteria
                - Enumerate all available options
                - If list of products are requested, then provide a list of products
                - Include any relevant cross-selling opportunities

                2. For specific queries:
                - Provide detailed, step-by-step information
                - Clearly identify responsible parties for each action
                - Include relevant timelines and prerequisites
                - Note any exceptions or special conditions
                - If list of products are requested, then provide a list of products

                Remember: If the information is not present in the provided context, acknowledge this limitation rather than making assumptions.

                Chat History:
                {chat_context}

                Retrieved Information:
                {doc_context}
                """
            print("System prompt is not provided. Using default prompt.")
        else:
            self.system_prompt = system_prompt

    def retrieve_documents(self, state: ChatState) -> ChatState:
        """Retrieve relevant documents from the vector store based on the query."""
        # Retrieve documents
        retrieved_docs = self.search_manager.search(state.current_query)

        # Update state with retrieved documents
        return ChatState(
            messages=state.messages,
            chat_history=state.chat_history,
            current_query=state.current_query,
            retrieved_documents=retrieved_docs
        )
    
    @staticmethod
    def format_content(state: ChatState) -> str:
        retrieved_documents = state.retrieved_documents[:]
        retrieved_documents.sort(key=lambda x: x['chunk_id'], reverse=False)
        doc_context = "\n\n".join([
            f"Document {doc['title']}:\n{doc['content']}"
            for i, doc in enumerate(state.retrieved_documents)
        ])
        return doc_context
    
    def generate_context(self, state: ChatState) -> ChatState:
        """Generate a context string from retrieved documents and chat history."""
        # Format retrieved documents
        doc_context = self.format_content(state)
        
        # Format the most recent messages (last 5 exchanges)
        recent_chat_history = state.chat_history[-5:] if state.chat_history else []
        chat_context = "\n".join([
            f"Human: {human}\nAI: {ai}"
            for human, ai in recent_chat_history
        ])
        
        # Create the context message
        context_message = {
            "role": "system",
            "content": self.system_prompt(chat_context=chat_context, doc_context=doc_context)
        }

        
        # Update the messages with the context
        updated_messages = [context_message] + state.messages
        
        return ChatState(
            messages=updated_messages,
            chat_history=state.chat_history,
            current_query=state.current_query,
            retrieved_documents=state.retrieved_documents
        )

    def format_llm_output(self, response: cited_answer) -> str:
        """Format the LLM output into a human-readable response."""
        # Format references as a list with proper newline handling
        sources = "\n\n**References**\n" + "\n".join(["- " + citation.strip() for citation in response.citations])
        # Format answer properly
        formatted_answer = response.answer.replace("\n\n", "\n").strip()
        return formatted_answer + sources
    
    def generate_response(self, state: ChatState) -> ChatState:
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
            response = self.chat_model.invoke(formatted_messages)
            # if not isinstance(response, cited_answer):
            #     raise ValueError("LLM response did not match the required cited_answer format")
            
            # response_text = self.format_llm_output(response)
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
            
            # Provide a fallback response that follows the required format
            fallback_response = cited_answer(
                answer="I apologize, but I encountered an error processing your request. Could you please rephrase your question?",
                citations=[]
            )
            response_text = self.format_llm_output(fallback_response)
            
            # Update chat history with fallback response
            updated_chat_history = state.chat_history + [(state.current_query, response_text)]
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

    def route_query(self, state: ChatState) -> ChatState:
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

    def should_retrieve(self, state: ChatState) -> str:
        """Determine if we should retrieve documents based on the state."""
        return state.should_retrieve

    def build_rag_chat_graph(self) -> StateGraph:
        """Create the LangGraph for our RAG chatbot."""
        self.workflow = StateGraph(ChatState)
        
        # Add nodes
        self.workflow.add_node("root", self.route_query)
        self.workflow.add_node("retrieve_documents", self.retrieve_documents)
        self.workflow.add_node("generate_context", self.generate_context)
        self.workflow.add_node("generate_response", self.generate_response)

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

    def process_query(self, query: str, chat_history: List[Tuple[str, str]] = None) -> Tuple[str, List[Tuple[str, str]]]:
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
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        # Extract the latest AI response
        latest_response = final_state['messages'][-1]['content']
        
        return latest_response, final_state['chat_history']
    
    def _get_css_styles(self) -> str:
        """Get the CSS styles for the chat interface."""
        return """
        <style>
        /* Style for chat message content */
        .stChatMessage div[data-testid="stMarkdownContainer"] {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            word-wrap: break-word;
            max-width: 100%;
            line-height: 1.5;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }
        .stChatMessage div[data-testid="stMarkdownContainer"] p {
            margin: 0 0 10px 0;
        }
        .stChatMessage div[data-testid="stMarkdownContainer"] ul {
            margin: 0;
            padding-left: 20px;
            list-style-position: outside;
            list-style-type: none;
        }
        .stChatMessage div[data-testid="stMarkdownContainer"] li {
            margin: 0;
            padding: 0;
            text-indent: -16px;
            padding-left: 16px;
            display: block;
            line-height: 1.4;
        }
        .stChatMessage div[data-testid="stMarkdownContainer"] li::before {
            content: "-";
            display: inline-block;
            width: 16px;
            margin-left: -4px;
        }
        .references-box {
            border-left: 3px solid #3d5afe;
            padding-left: 20px;
            margin: 20px 0;
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
        }
        .compact-list {
            margin: 0;
            padding-left: 20px;
            list-style-position: outside;
            list-style-type: none;
        }
        .compact-list li {
            margin: 0;
            padding: 0;
            text-indent: -16px;
            padding-left: 16px;
            display: block;
            line-height: 1.4;
        }
        .compact-list li::before {
            content: "-";
            display: inline-block;
            width: 16px;
            margin-left: -4px;
        }
        .chat-title {
            text-align: center;
            color: #1e88e5;
            margin-bottom: 30px;
        }
        /* Style for example buttons */
        div[data-testid="stHorizontalBlock"] button {
            width: 100%;
            background-color: #f0f2f6;
            border: 1px solid #e0e3e9;
            border-radius: 8px;
            color: #1e88e5;
            transition: all 0.3s;
            margin: 4px 0;
            padding: 8px;
            min-height: 0;
        }
        div[data-testid="stHorizontalBlock"] button:hover {
            background-color: #e3f2fd;
            border-color: #1e88e5;
        }
        .examples-title {
            text-align: center;
            margin: 10px 0;
            font-weight: bold;
            color: #1e88e5;
        }
        </style>
        """

    def _render_references(self, references: str) -> None:
        """Render references section with proper formatting."""
        references_list = [ref.strip() for ref in references.split('\n') if ref.strip()]
        references_html = ''.join([f"<li>{ref}</li>" for ref in references_list])
        st.markdown(f"""
        <div class='references-box'>
            <strong>📚 References</strong>
            <ul class='compact-list'>{references_html}</ul>
        </div>
        """, unsafe_allow_html=True)

    def _render_message(self, content: str) -> None:
        """Render a chat message with proper formatting."""
        content_parts = content.split("\n\n**References**\n")
        if len(content_parts) > 1:
            answer, references = content_parts
            with st.container():
                st.markdown(answer)
            self._render_references(references)
        else:
            st.markdown(content)

    def run_demo(self, chat_historyIn: List[Tuple[str, str]] = []):
        def respond(message, chat_historyIn):
            """Process the user message and update the chat history."""
            
            # Process the query
            response, chat_historyIn = self.process_query(message, chat_historyIn)
            
            return response
        
        demo = gr.ChatInterface(
            fn=respond,
            title="Athena Chatbot with LangGraph",
            description="Ask questions about documents in your knowledge base or have a general conversation.",
            examples=[
                "Ποια είναι τα καταναλωτικά δάνεια που δίνουμε?",
                "ποια είναι τα βήματα της διαδικασίας για τη μεταβολή της διάρκειας ενός στεγαστικού δανείου;",
                "γράψε μου τα βασικά βήματα για αίτημα Άρσης Βάρους στα στεγαστικά",
                "Μεγιστο οριο ηλικιας για στεγαστικο δανειο?"
            ],
            theme="soft"
        )
        demo.launch(share=True)

    def run_demo2(self, chat_historyIn: List[Tuple[str, str]] = []):
        def handle_message(prompt: str, chat_history: List[Tuple[str, str]]) -> None:
            """Handle a new message from the user."""
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Show processing indicator
            with st.spinner('Thinking...'):
                response, _ = self.process_query(prompt, chat_history)
                
                # Display assistant response
                with st.chat_message("assistant"):
                    self._render_message(response)

            st.session_state.messages.append({"role": "assistant", "content": response})

        import streamlit as st
        
        # Set page config and apply styles
        st.set_page_config(
            page_title="NBG Athena Chatbot",
            page_icon="🤖",
            layout="wide"
        )
        st.markdown(self._get_css_styles(), unsafe_allow_html=True)

        # Display title and description
        st.markdown("<h1 class='chat-title'>NBG Athena Chatbot</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-size: 1.2em;'>Ask questions about documents in your knowledge base or have a general conversation.</p>", unsafe_allow_html=True)
        st.divider()

        # Initialize chat history in session state if not already present
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display example questions as clickable buttons
        examples = [
            "Ποια είναι τα καταναλωτικά δάνεια που δίνουμε?",
            "ποια είναι τα βήματα της διαδικασίας για τη μεταβολή της διάρκειας ενός στεγαστικού δανείου;",
            "γράψε μου τα βασικά βήματα για αίτημα Άρσης Βάρους στα στεγαστικά",
            "Μεγιστο οριο ηλικιας για στεγαστικο δανειο?"
        ]
        
        st.markdown("<div class='examples-title'>Try these examples</div>", unsafe_allow_html=True)
        # Create columns for better layout
        cols = st.columns([1] * 2)  # 2 columns
        for i, example in enumerate(examples):
            with cols[i % 2]:
                if st.button(example, key=f"example_{i}", help="Click to use this example"):
                    chat_history = [(msg["content"], st.session_state.messages[i+1]["content"])
                                 for i, msg in enumerate(st.session_state.messages[:-1:2])]
                    handle_message(example, chat_history)
                    st.rerun()
        st.divider()

        # Display existing chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    self._render_message(message["content"])
                else:
                    st.markdown(message["content"])

        # Handle user input
        if prompt := st.chat_input("Enter your message"):
            chat_history = [(msg["content"], st.session_state.messages[i+1]["content"])
                          for i, msg in enumerate(st.session_state.messages[:-1:2])]
            handle_message(prompt, chat_history)

    def run(self):
        # Simple chat loop
        chat_history = []
        print("RAG Chatbot initialized. Type 'exit' to quit.")
        
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == "exit":
                break
            
            response, chat_history = self.process_query(user_input, chat_history)
            print(f"\nChatbot: {response}")