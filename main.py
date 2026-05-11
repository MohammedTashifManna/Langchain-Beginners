import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
load_dotenv()

data_path =  'product_reviews.csv'
df = pd.read_csv(data_path)
df.head()

llm_model="llama-3.3-70b-versatile"
llm = ChatGroq(temperature=0.9, model=llm_model)

product=df.Product[1]

first_prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe a company that makes {product}?"
)
second_prompt = ChatPromptTemplate.from_template(
    "Write a 30 words description for the following company:{company_name}"
)
format_parsed_output = RunnableLambda(lambda output: {"company_name": output})
# Chain 1
chain = first_prompt|llm|format_parsed_output|second_prompt|llm
chain_ans=chain.invoke({"product":product})
print(chain_ans.content)

first_prompt = ChatPromptTemplate.from_template(
    "Translate the following review to english:"
    "\n\n{Review}"
)

second_prompt = ChatPromptTemplate.from_template(
    "Can you summarize the following review in 1 sentence:"
    "\n\n{English_Review}"
)

third_prompt = ChatPromptTemplate.from_template(
    "What language is the following review:\n\n{Review}"
)

fourth_prompt = ChatPromptTemplate.from_template(
    "Write a follow up response to the following "
    "summary in the specified language:"
    "\n\nSummary: {summary}\n\nLanguage: {language}"
)

formated_chain_one=RunnableLambda(lambda output: {"English_Review": output})
formated_chain_two=RunnableLambda(lambda output: {"summary": output})
fchain_one=first_prompt|llm|formated_chain_one|second_prompt|llm
review = df.Review[1]
fchainout=fchain_one.invoke(review)
fchainoutContent=fchainout.content
print(fchainout.content)
chain_three=third_prompt|llm
chainout=chain_three.invoke(review)
chainoutContent=chainout.content
print(chainout.content)

chain_four=fourth_prompt|llm
response = chain_four.invoke(
    {
        "summary":fchainoutContent,
        "language":chainoutContent
    }
)

print(response.content)

parser = StrOutputParser()

physics_template = """
You are a very smart physics professor.

Question:
{input}
"""

math_template = """
You are a very good mathematician.

Question:
{input}
"""

history_template = """
You are a very good historian.

Question:
{input}
"""

computerscience_template = """
You are a successful computer scientist.

Question:
{input}
"""

physics_chain = (
    ChatPromptTemplate.from_template(physics_template)
    | llm
    | parser
)

math_chain = (
    ChatPromptTemplate.from_template(math_template)
    | llm
    | parser
)

history_chain = (
    ChatPromptTemplate.from_template(history_template)
    | llm
    | parser
)

cs_chain = (
    ChatPromptTemplate.from_template(computerscience_template)
    | llm
    | parser
)

default_chain = (
    ChatPromptTemplate.from_template("{input}")
    | llm
    | parser
)

router_template = """
You are a router.

Decide which subject best matches the user's question.

Possible destinations:
- physics
- math
- history
- computer_science
- default

Only return the destination name.

Question:
{input}
"""

router_prompt = ChatPromptTemplate.from_template(router_template)

router_chain = (
    router_prompt
    | llm
    | parser
)


def route(info):
    topic = info["topic"].strip().lower()

    if "physics" in topic:
        return physics_chain

    elif "math" in topic:
        return math_chain

    elif "history" in topic:
        return history_chain

    elif "computer_science" in topic:
        return cs_chain

    else:
        return default_chain

full_chain = (
    {
        "topic": router_chain,
        "input": lambda x: x["input"]
    }
    | RunnableLambda(route)
)

response = full_chain.invoke({
    "input": "What is Generative AI?"
})

print(response)
