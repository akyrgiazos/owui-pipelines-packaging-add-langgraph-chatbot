# Azure Search Configuration
import os
from dotenv import load_dotenv
load_dotenv()

AZURE_TENANT_ID=os.getenv("AZURE_TENANT_ID")
AZURE_CLIENT_ID=os.getenv("AZURE_CLIENT_ID")
AZURE_CLIENT_SECRET=os.getenv("AZURE_CLIENT_SECRET")
AZURE_SUBSCRIPTION_ID=os.getenv("AZURE_SUBSCRIPTION_ID")
AZURE_RESOURCE_GROUP=os.getenv("AZURE_RESOURCE_GROUP_NAME") 


AZURE_STORAGE_ACCOUNT_NAME=os.getenv("AZURE_STORAGE_ACCOUNT_NAME") #blob storage account name
AZURE_STORAGE_CONTAINER_NAME=os.getenv("AZURE_STORAGE_CONTAINER_NAME") #blob storage container name
AZURE_SEARCH_SERVICE_NAME=os.getenv("AZURE_SEARCH_SERVICE_NAME") #AI search service name
AZURE_INDEX_NAME=os.getenv("AZURE_INDEX_NAME") #index name in azure AI search service
AZURE_INDEX_SEMANTIC_CONFIGURATION=os.getenv("AZURE_INDEX_SEMANTIC_CONFIGURATION") #semantic configuration name in azure AI search service

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_INDEX_NAME")
AZURE_SEARCH_API_VERSION = os.getenv("AZURE_SEARCH_API_VERSION")


# Azure OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_EMBEDDING_MODEL= os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
AZURE_OPENAI_CHAT_MODEL = os.getenv("AZURE_OPENAI_CHAT_MODEL")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_SEARCH_SERVICE_NAME = os.getenv("AZURE_SEARCH_SERVICE_NAME")

#Domains
DOMAINS = os.getenv("SEARCH_DOMAINS")
if DOMAINS is not None:
    DOMAINS = DOMAINS.split(",")
else:
    DOMAINS = []    

VECTOR_FIELDS = os.getenv("VECTOR_FIELDS")
# Logging Configuration
LOG_FILE = os.path.join(os.path.dirname(__file__), 'logs', 'azure_search.log')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'
