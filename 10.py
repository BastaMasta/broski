from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.llms import Cohere

ipc_file_path = "penal.txt"

with open(ipc_file_path, "r", encoding="utf-8") as file:
    ipc_text = file.read()

ipc_document = Document(page_content=ipc_text)

llm = Cohere(cohere_api_key="Hw6tu5eTo6msCsY0DWSWf9ceu1PrvBxDEvXhVQrV")

qa_chain = load_qa_chain(llm, chain_type="stuff")

print("Chatbot for the Indian Penal Code (IPC)")
print("Ask a question about the Indian Penal Code (type 'exit' to stop):")
while True:
    user_question = input("\nYour question: ")
    if user_question.lower() == "exit":
        print("Goodbye!")
        break

    response = qa_chain.run(input_documents=[ipc_document], question=user_question)
    print(f"Answer: {response}")