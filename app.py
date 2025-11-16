from flask import Flask, render_template, request , jsonify
from langchain_pinecone import PineconeVectorStore
from src.helper import download_embeddings

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


import os
from src.prompt import system_prompt
from src.helper import download_embeddings

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY2 = os.environ.get("OPENAI_API_KEY2")


embeddings = download_embeddings()

index_name = "medxgen-ayurvedic"
docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=index_name,
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 2})

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.4,
    openai_api_key=OPENAI_API_KEY2
)


prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(
    llm,
    prompt,
    document_variable_name="context"   # <── FIX
)

rag_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=question_answer_chain
)












# prompt = ChatPromptTemplate.from_messages([
#     ("system", system_prompt),
#     ("human", "{input}"),
# ])

# question_answer_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def bot_ui():
    return render_template("bot.html")

@app.route("/get", methods=["POST"])
def bot():
    msg = request.form["msg"]
    print("User Message:", msg)
    response = rag_chain.invoke({"input": msg})
    # print("Bot Response:", response["answer"])
    return str(response["answer"])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
