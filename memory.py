from pymongo import MongoClient
from langchain.memory import ConversationBufferMemory, MongoDBChatMessageHistory
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import langchain

langchain.debug = True

client = MongoClient("mongodb://localhost:27017/")
db = client["chat_memory"]
collection = db["conversations"]
model = "llama3.2"


template = """You are a helpful assistant.
Here is our conversation history:
{history}
User: {input}
Assistant:"""

prompt = PromptTemplate(input_variables=["history", "input"], template=template)


class MongoMemory:

        def __init__(self, collection):
            self.collection = collection

        def save_message(self, user_input, bot_response):
            self.collection.insert_one({"user": user_input, "bot": bot_response})

        def load_messages(self):
            messages = self.collection.find({}, {"_id": 0})
            return [f"User: {m['user']}\nBot: {m['bot']}" for m in messages]


# Initialize MongoDB memory
mongo_memory = MongoMemory(collection)

# LangChain Memory
memory = ConversationBufferMemory()

print(memory.chat_memory.messages)

previous_messages = mongo_memory.load_messages()
for msg in previous_messages:
    parts = msg.split("\n")
    memory.chat_memory.add_user_message(parts[0].replace("User: ", ""))
    memory.chat_memory.add_ai_message(parts[1].replace("Bot: ", ""))

conversation = LLMChain(llm=ChatOllama(model=model), memory=memory, prompt=prompt)

# Example conversation flow
user_input = "Tell me about sustainability in Artificial Intelligence"
bot_response = conversation.invoke(user_input)

# Store in MongoDB
mongo_memory.save_message(user_input, bot_response)

# Retrieve all messages
chat_history = mongo_memory.load_messages()
print("\n".join(chat_history))