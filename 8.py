from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain import LLMChain
from google.colab import drive

drive.mount('/content/drive')
file_path = "/content/drive/MyDrive/rbmk.txt"

with open(file_path, "r") as file:
    text = file.read()

cohere_api_key = "API_KEY_HERE"

prompt_template = """
Summarize the following text in two bullet points:
{text}
"""

llm = Cohere(cohere_api_key=cohere_api_key)
prompt = PromptTemplate(input_variables=["text"], template=prompt_template)

chain = LLMChain(llm=llm, prompt=prompt)

result = chain.run(text)
print(text)

print("Summarized Output in Bullet Points:")
print(result)