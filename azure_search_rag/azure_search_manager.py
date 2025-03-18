from enum import Enum
from typing import List, Dict, Any, Callable, Optional
from azure_search_rag.logger import log, INFO, ERROR, WARNING
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.chat_models.base import BaseChatModel
import time
from azure_search_rag.wrapper import Identity
from langchain.schema import Document
import requests
from tqdm import tqdm
from azure_search_rag.config import (
    AZURE_OPENAI_CHAT_MODEL,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_EMBEDDING_MODEL,
    AZURE_OPENAI_KEY,
    AZURE_TENANT_ID,
    AZURE_CLIENT_ID,
    AZURE_CLIENT_SECRET,
    AZURE_SUBSCRIPTION_ID,
    AZURE_RESOURCE_GROUP,
    AZURE_SEARCH_SERVICE_NAME,
    AZURE_SEARCH_INDEX_NAME,
    AZURE_INDEX_SEMANTIC_CONFIGURATION,
    VECTOR_FIELDS,
    DOMAINS
)

class AzureSearchManager:
    # ΟΚ here i pass the Identity object to the AzureSearchManager. Another option would be to pass service name or index_name.
    identity: Identity
    embed_document: Callable[[List[str]], List[List[float]]]
    llm: BaseChatModel

    def __init__(self, 
                 identity: Identity, 
                 embed_document: Callable[[List[str]], List[List[float]]],
                 llm: BaseChatModel):
        self.embed_document = embed_document
        self.llm = llm
        self.identity = identity
        self.subscription = self.identity.get_subscription(subscription_id=AZURE_SUBSCRIPTION_ID)
        self.resource_group = self.subscription.get_resource_group(AZURE_RESOURCE_GROUP)
        self.search_service = self.subscription.get_search_service(AZURE_SEARCH_SERVICE_NAME)
        if self.search_service is None:
            raise ValueError(f"Search service '{AZURE_SEARCH_SERVICE_NAME}' not found in subscription '{AZURE_SUBSCRIPTION_ID}'")

        self.index =  self.search_service.get_index(AZURE_SEARCH_INDEX_NAME)
        self.semantic_config_name = AZURE_INDEX_SEMANTIC_CONFIGURATION
        self.vector_fields = VECTOR_FIELDS
        self.domains = DOMAINS

    def generate_embeddings(self, documents: List[Document]) -> bool:
        """
        Generate embeddings for a list of documents
        """
        try:
            embeddings = [ self.embed_document([doc['content']])[0] for doc in tqdm(documents, desc="Generating embeddings")]
            for doc, emb in zip(documents, embeddings):
                doc['content_vector'] = emb
            return True
        except Exception as e:
            log(ERROR, f"Failed to generate embeddings: {str(e)}")
            return False
    
    def category_prompt(self) -> PromptTemplate:
        # Create prompt template
        from prompts import category_prompt_template
        return PromptTemplate(template=category_prompt_template, input_variables=['domains','question'])
    
    def category_chain(self, question: str) -> str:
        """
        Get the category of the question
        """
        chain = self.category_prompt() | self.llm | StrOutputParser()
        result = chain.invoke({"question": question, "domains": ",".join(self.domains)})
        return result 
    
    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        try:
            type_filter = self.category_chain(query)
            if self.semantic_config_name is not None:
                results_all = self.index.search_with_context_window(
                                query_text=query,
                                query_vector=self.embed_document([query])[0],
                                vector_fields=self.vector_fields,
                                use_semantic_search=True,
                                semantic_config_name=self.semantic_config_name,
                                window_size=3,  # 3 chunks before and after
                                top=top_k,
                                # We need to make a field in the index as filterable apart from the parent_id field.
                                search_options={"filter": f"search.in('*{type_filter.upper()}*', 'readable_url', 'full', 'any')"}
                                )
            else:
                results_all = self.index.search_with_context_window(
                                query_text=query,
                                query_vector=self.embed_document([query])[0],
                                vector_fields=self.vector_fields,
                                use_semantic_search=False,
                                window_size=3,  # 3 chunks before and after
                                top=top_k,
                                # We need to make a field in the index as filterable apart from the parent_id field.
                                search_options={"filter": f"search.in('*{type_filter.upper()}*', 'readable_url', 'full', 'any')"}
                                )

            # Format results
            search_results = []
            for doc in results_all:
                result = {
                    'content': doc['chunk'],
                    'title': doc['title'],
                    'url': doc['url'],
                    'chunk_id':doc['chunk_id']
                }
                search_results.append(result)
            # log(INFO, f"Search completed successfully - Found {len(search_results)} results")
            return search_results

        except Exception as e:
            log(ERROR, f"Search error: {str(e)}")
            return []
        

def initialize_azure_search() -> AzureSearchManager:
    """Initialize Azure Search with OpenAI embeddings"""
    try:
        log(INFO, "Initializing text embeddings...")

        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_KEY,
            model=AZURE_OPENAI_EMBEDDING_MODEL
        )

        try:
            embeddings.embed_query("test")
        except Exception as e:
            error_msg = ( f"OpenAI embeddings failed. Please verify your OpenAI API key. Error: {str(e)}" )
            log(ERROR, error_msg)
            raise ValueError(error_msg) 

        def embed_document(doc: List[str]) -> List[List[float]]:
            return embeddings.embed_documents(doc)

        llm = AzureChatOpenAI(
            model=AZURE_OPENAI_CHAT_MODEL,
            temperature=0.0,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_KEY,
        )

        log(INFO, f"Successfully initialized AzureSearchManager with endpoint: {AZURE_SEARCH_SERVICE_NAME}")
        time.sleep(1)
        identity = Identity(AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET)
        time.sleep(1)
        manager = AzureSearchManager(identity, embed_document, llm)
        return manager

    except requests.exceptions.RequestException as e:
        error_msg = f"Network error connecting to Azure Search: {str(e)}"
        log(ERROR, error_msg)
        raise ValueError(error_msg)
    except Exception as e:
        log(ERROR, f"Error initializing AzureSearchManager: {str(e)}")
        raise

if __name__ == "__main__":
    manager= initialize_azure_search()
