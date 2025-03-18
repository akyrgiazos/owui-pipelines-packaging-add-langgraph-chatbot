from pathlib import Path
from typing import List, Union, Generator, Iterator, Dict, Tuple, Any, Optional
import os
from pydantic import BaseModel

from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.schema import Document
from azure_search_rag.wrapper import Identity
from azure_search_rag.rag_chain import RAGChain

class Pipeline:
    """Pipeline implementation that uses RAG chain from azure-search-rag."""
    
    class Valves(BaseModel):
        """Options to change from the WebUI"""
        TENANT_ID: str = ""
        CLIENT_ID: str = ""
        CLIENT_SECRET: str = ""
        AZURE_OPENAI_CHAT_MODEL: str = ""
        AZURE_OPENAI_API_VERSION: str = ""
        AZURE_OPENAI_ENDPOINT: str = ""
        AZURE_OPENAI_EMBEDDING_MODEL: str = ""
        AZURE_OPENAI_KEY: str = ""
        AZURE_SUBSCRIPTION_ID: str = ""
        AZURE_RESOURCE_GROUP: str = ""
        AZURE_SEARCH_SERVICE_NAME: str = ""
        AZURE_SEARCH_INDEX_NAME: str = ""
        AZURE_INDEX_SEMANTIC_CONFIGURATION: str = ""
        VECTOR_FIELDS: str = ""
        DOMAINS: str = ""
        VERBOSE_TRACE: bool = False

    def __init__(self):
        self.chat_history = {}
        self.name = "Azure RAG Search"
        self._previous_valves = None
        
        # Initialize valves with environment variables
        self.valves = self.Valves(**{
            "TENANT_ID": os.getenv("AZURE_TENANT_ID", ""),
            "CLIENT_ID": os.getenv("AZURE_CLIENT_ID", ""),
            "CLIENT_SECRET": os.getenv("AZURE_CLIENT_SECRET", ""),
            "AZURE_OPENAI_CHAT_MODEL": os.getenv("AZURE_OPENAI_CHAT_MODEL", ""),
            "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION", ""),
            "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            "AZURE_OPENAI_EMBEDDING_MODEL": os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", ""),
            "AZURE_OPENAI_KEY": os.getenv("AZURE_OPENAI_KEY", ""),
            "AZURE_SUBSCRIPTION_ID": os.getenv("AZURE_SUBSCRIPTION_ID", ""),
            "AZURE_RESOURCE_GROUP": os.getenv("AZURE_RESOURCE_GROUP_NAME", ""),
            "AZURE_SEARCH_SERVICE_NAME": os.getenv("AZURE_SEARCH_SERVICE_NAME", ""),
            "AZURE_SEARCH_INDEX_NAME": os.getenv("AZURE_INDEX_NAME", ""),
            "AZURE_INDEX_SEMANTIC_CONFIGURATION": os.getenv("AZURE_INDEX_SEMANTIC_CONFIGURATION", ""),
            "VECTOR_FIELDS": os.getenv("VECTOR_FIELDS", ""),
            "DOMAINS": os.getenv("SEARCH_DOMAINS", ""),
            "VERBOSE_TRACE": False,
        })
        
        # Store initial values for comparison
        self._previous_valves = self.valves.dict()
        
        # Initialize the RAG components
        try:
            self.embeddings = self.initialize_embeddings()
            self.llm = self.initialize_llm()
            self.search_manager = self.initialize_azure_search()
            self.rag_chain = RAGChain(search_manager=self.search_manager, chat_model=self.llm)
        except Exception as e:
            print(f"Error initializing RAG components: {str(e)}")
            self.search_manager = None
            self.llm = None
            self.rag_chain = None
            
    def initialize_embeddings(self):
        """Initialize Azure OpenAI embeddings using valve values instead of config"""
        try:
            embeddings = AzureOpenAIEmbeddings(
                azure_endpoint=self.valves.AZURE_OPENAI_ENDPOINT,
                api_key=self.valves.AZURE_OPENAI_KEY,
                model=self.valves.AZURE_OPENAI_EMBEDDING_MODEL
            )
            # Test embeddings
            embeddings.embed_query("test")
            return embeddings
        except Exception as e:
            print(f"OpenAI embeddings failed. Error: {str(e)}")
            raise ValueError(f"OpenAI embeddings failed. Error: {str(e)}")
    
    def embed_document(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts"""
        return self.embeddings.embed_documents(texts)
    
    def initialize_llm(self) -> BaseChatModel:
        """Initialize Azure OpenAI LLM using valve values instead of config"""
        return AzureChatOpenAI(
            model=self.valves.AZURE_OPENAI_CHAT_MODEL,
            temperature=0.0,
            api_version=self.valves.AZURE_OPENAI_API_VERSION,
            azure_endpoint=self.valves.AZURE_OPENAI_ENDPOINT,
            api_key=self.valves.AZURE_OPENAI_KEY,
        )
        
    def initialize_azure_search(self):
        """Initialize Azure Search manager using valve values instead of config"""
        from azure_search_rag.azure_search_manager import AzureSearchManager
        
        identity = Identity(
            self.valves.TENANT_ID, 
            self.valves.CLIENT_ID, 
            self.valves.CLIENT_SECRET
        )
        
        # Create search manager with local embed_document method that uses our embeddings
        manager = AzureSearchManager(
            identity=identity,
            embed_document=self.embed_document,
            llm=self.llm
        )
        
        # Override config-loaded values with valve values
        manager.subscription = identity.get_subscription(subscription_id=self.valves.AZURE_SUBSCRIPTION_ID)
        manager.resource_group = manager.subscription.get_resource_group(self.valves.AZURE_RESOURCE_GROUP)
        manager.search_service = manager.subscription.get_search_service(self.valves.AZURE_SEARCH_SERVICE_NAME)
        
        if manager.search_service is None:
            raise ValueError(f"Search service '{self.valves.AZURE_SEARCH_SERVICE_NAME}' not found")
            
        manager.index = manager.search_service.get_index(self.valves.AZURE_SEARCH_INDEX_NAME)
        manager.semantic_config_name = self.valves.AZURE_INDEX_SEMANTIC_CONFIGURATION
        manager.vector_fields = self.valves.VECTOR_FIELDS
        
        # Split domains string if it exists
        domains = []
        if self.valves.DOMAINS:
            domains = self.valves.DOMAINS.split(",")
        manager.domains = domains
        
        return manager

    def _check_for_config_changes(self) -> bool:
        """Check if any valve values have changed and need reinitialization"""
        current_valves = self.valves.dict()
        
        # Check if any configuration values have changed
        changed = False
        changed_values = []
        
        for key, value in current_valves.items():
            # Skip verbose trace as it doesn't affect components
            if key == "VERBOSE_TRACE":
                continue
                
            if self._previous_valves.get(key) != value:
                changed = True
                changed_values.append(key)
                
        # Update previous valves if changes were detected
        if changed:
            print(f"Configuration values changed: {', '.join(changed_values)}")
            self._previous_valves = current_valves
        
        return changed
    
    def _reinitialize_components(self):
        """Reinitialize components when configuration changes"""
        print("Reinitializing RAG components due to configuration changes...")
        try:
            self.embeddings = self.initialize_embeddings()
            self.llm = self.initialize_llm()
            self.search_manager = self.initialize_azure_search()
            self.rag_chain = RAGChain(search_manager=self.search_manager, chat_model=self.llm)
            print("RAG components successfully reinitialized")
            return True
        except Exception as e:
            print(f"Error reinitializing RAG components: {str(e)}")
            self.search_manager = None
            self.llm = None
            self.rag_chain = None
            return False
        
    def pipe(
            self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[Iterator[str], str]:
        """Process a user message and return a response."""
        
        # Skip processing for specific conditions
        if not self.valves.VERBOSE_TRACE:
            if "metadata" in body and body["metadata"] is not None and "task" in body["metadata"]:
                return ""

        print(f"pipe:{__name__} running")
        
        if "metadata" in body and body["metadata"] is not None:
            if "task" in body["metadata"]:
                return ""
        
        # Check if any configuration has changed and reinitialize if needed
        if self._check_for_config_changes():
            self._reinitialize_components()
        
        # Get user ID for tracking chat history
        chat_id = body['user']['id']
        
        # Check if RAG components are initialized
        if not self.rag_chain or not self.search_manager or not self.llm:
            return "Azure RAG Search components are not initialized. Please check the configuration and try again."
        
        # Get or initialize chat history
        if chat_id not in self.chat_history:
            self.chat_history[chat_id] = []
        
        # Process the query through the RAG chain
        if user_message.lower() == "delete":
            # Clear chat history
            self.chat_history.pop(chat_id, None)
            return "Chat history cleared."
        
        try:
            # Process the query using the RAG chain
            response, updated_chat_history = self.rag_chain.process_query(
                user_message, 
                self.chat_history.get(chat_id, [])
            )
            
            # Update the chat history
            self.chat_history[chat_id] = updated_chat_history
            
            return response
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            
            print(f"Error processing query: {str(e)}")
            print(f"Traceback: {error_trace}")
            
            # Provide a more helpful error message if it's a configuration issue
            if "azure.core.exceptions" in error_trace or "401" in error_trace:
                return "Authentication error with Azure services. Please check your Azure credentials and try again."
            elif "openai" in error_trace.lower():
                return "Error with OpenAI API. Please check your OpenAI API key and settings."
            elif "search" in error_trace.lower() and "index" in error_trace.lower():
                return "Error with Azure Search. Please verify your search service and index configurations."
            else:
                return f"Sorry, I encountered an error while processing your query: {str(e)}"