from pathlib import Path
from typing import List, Union, Generator, Iterator
import uuid

from pydantic import BaseModel

import os, time
from azure.identity import ClientSecretCredential
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.projects.models import CodeInterpreterTool
from azure.ai.projects.models import (
    OpenAIPageableListOfThreadMessage, 
    ThreadMessage, 
    MessageContent,
    MessageTextUrlCitationAnnotation,
    MessageTextFileCitationAnnotation,
    MessageRole
)

def get_text_from_thread_message(message:ThreadMessage):
    content: List[MessageContent] = message.content
    for content_item in content:
        if content_item.type == "text":
            initial_text = content_item.text["value"]
            text = content_item.text["value"]
            if "annotations" in content_item.text:
                annotations = content_item.text["annotations"]
                for annotation in annotations:
                    if annotation.type == "url_citation":
                        annotation_text = annotation.text
                        url_citation = annotation.url_citation
                        if url_citation is not None:
                            annotation_title = url_citation.title
                            annotation_url = url_citation.url
                            if (annotation_url != annotation_title): 
                                text = text.replace(annotation_text, f"\n[{annotation_title}]({annotation_url})")
                            else:
                                text = text.replace(annotation_text, f"\n[{annotation_title}]")
                    elif annotation.type == "file_citation":
                        file_citation = annotation.file_citation
            if initial_text != text:
                content_item.text["value"] = text
            return text 
    return ""

class Pipeline:
    class Valves(BaseModel):
        """Options to change from the WebUI"""
        AZURE_CONNECTION_STRING: str = ""
        TENANT_ID: str = ""
        CLIENT_ID: str = ""
        CLIENT_SECRET: str = ""
        AZURE_AGENT_ID: str = ""
        VERBOSE_TRACE: bool = False

    def __init__(self):
        self.threads = {}
        self.name = "Azure Agent"
        self.valves = self.Valves(**{
            "AZURE_CONNECTION_STRING": os.getenv("AZURE_CONNECTION_STRING", ""),
            "TENANT_ID": os.getenv("AZURE_TENANT_ID", ""),
            "CLIENT_ID": os.getenv("AZURE_CLIENT_ID", ""),
            "CLIENT_SECRET": os.getenv("AZURE_CLIENT_SECRET", ""),
            "AZURE_AGENT_ID": os.getenv("AZURE_AGENT_ID", ""),
        })
        
        if ( not self.valves.TENANT_ID or not self.valves.CLIENT_ID or not self.valves.CLIENT_SECRET):
            self.credential = None
            self.project_client = None
        else:
            self.credential = ClientSecretCredential(self.valves.TENANT_ID, self.valves.CLIENT_ID, self.valves.CLIENT_SECRET)
            self.project_client = AIProjectClient.from_connection_string(
                credential=self.credential, conn_str=self.valves.AZURE_CONNECTION_STRING
            )

    def pipe(
            self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[Iterator[str], str]:
        
        files = {}
        file_ids = []
        
        # to keep the log clean
        # remove for debugging
        if not self.valves.VERBOSE_TRACE:
            if "metadata" in body and body["metadata"] is not None and "task" in body["metadata"]:
                return ""   

        print(f"pipe:{__name__} running")
        print("BODY: ", body)
        if "messages" in body and body["messages"] is not None:
            print("USER REQUEST MESSAGES: ", body["messages"])
        if "user" in body and body["user"] is not None:
            print("USER: ", body["user"])
        if "metadata" in body and body["metadata"] is not None:
            if "files" in body["metadata"] and body["metadata"]["files"] is not None:
                for file in body["metadata"]["files"]:
                    if "type" in file and file["type"] == "file" and "file" in file:
                        file_obj = file["file"]
                        if "id" in file_obj:
                            file_ids.append(file_obj["id"])
                            files[file_obj["id"]] = file_obj
        print("USER_MESSAGE: ", user_message)

        if 'task' in body['metadata']:
            return ""

        chat_id = body['user']['id']

        if not self.project_client:
            if ( not self.valves.TENANT_ID or not self.valves.CLIENT_ID or not self.valves.CLIENT_SECRET):
                return "Please provide Azure credentials (TENANT_ID, CLIENT_ID, CLIENT_SECRET) in the environment variables"
            self.credential = ClientSecretCredential(self.valves.TENANT_ID, self.valves.CLIENT_ID, self.valves.CLIENT_SECRET)
            if ( not self.valves.AZURE_CONNECTION_STRING):
                return "Please provide Azure connection string in the environment variables" 
            self.project_client = AIProjectClient.from_connection_string(
                credential=self.credential, conn_str=self.valves.AZURE_CONNECTION_STRING
            )

        if chat_id not in self.threads:
            thread = self.project_client.agents.create_thread()
            self.threads[chat_id] = { "thread_id": thread.id, "chat_id": chat_id, "files": [] }
            print(f"Created thread, thread ID: {thread.id}")

        else:
            chat = self.threads[chat_id]
            thread = self.project_client.agents.get_thread(chat["thread_id"])
            print(f"Using existing thread, thread ID: {thread.id}")

            if user_message == "Delete":
                print("Deleting thread")
                self.project_client.agents.delete_thread(thread.id)
                self.threads.pop(chat_id)
                return "Thread deleted"

        # Create a message
        message = self.project_client.agents.create_message(
            thread_id=thread.id,
            role="user",
            content=user_message,

        )
        print(f"Created message, message ID: {message.id}")

        # Run the agent
        run = self.project_client.agents.create_and_process_run(thread_id=thread.id, agent_id=self.valves.AZURE_AGENT_ID)
        print(f"Run finished with status: {run.status}")

        if run.status == "failed":
            # Check if you got "Rate limit is exceeded.", then you want to get more quota
            print(f"Run failed: {run.last_error}")

        # Get messages from the thread
        messages:OpenAIPageableListOfThreadMessage = self.project_client.agents.list_messages(thread_id=thread.id)
        print(f"AGENT RESPONSE MESSAGES: {messages}")

        # Get the last message from the sender
        message: ThreadMessage = messages.get_last_message_by_role(MessageRole.AGENT)
        
        return ( get_text_from_thread_message(message) )
