import asyncio
from typing import List, Tuple
from azure_search_manager import AzureSearchManager, initialize_azure_search
from azure_search_rag.rag_chain import RAGChain
import argparse
from tqdm import tqdm
from langchain.chat_models.base import BaseChatModel
from langchain_openai import AzureChatOpenAI
from langchain.chat_models.base import BaseChatModel
from config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_CHAT_MODEL,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_KEY,
)

search_manager: AzureSearchManager
llm: BaseChatModel 

async def chat_loop(search_manager: AzureSearchManager, llm: BaseChatModel):
    """
    Interactive chat loop using the RAG system
    """
    from prompts import system_prompt_template

    rag_chain = RAGChain(search_manager=search_manager, chat_model=llm, system_prompt=system_prompt_template)   
    await rag_chain.run()

def initialize():
    search_manager = initialize_azure_search()
    return search_manager

    
def main():
    parser = argparse.ArgumentParser(description="RAG System with Azure AI Search")
    
    args = parser.parse_args()
    search_manager = initialize_azure_search()

    llm = AzureChatOpenAI(
        model=AZURE_OPENAI_CHAT_MODEL,
        temperature=0.0,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
    )
    asyncio.run(chat_loop(search_manager=search_manager, llm=llm))

if __name__ == "__main__":
    main()