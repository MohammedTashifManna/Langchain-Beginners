import os
import groq
import tiktoken
from pinecone import Pinecone, ServerlessSpec
import tempfile
import pandas as pd
import yt_dlp
from dotenv import load_dotenv
from langchain_classic.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from uuid import uuid4
from langchain_core.prompts import PromptTemplate

load_dotenv()


YOUTUBE_VIDEOS = ["https://www.youtube.com/watch?v=vwlRzUddey4"]
client = groq.Groq()
def transcribe(youtube_url):

    with tempfile.TemporaryDirectory() as tmpdir:

        # File path template
        output_path = os.path.join(tmpdir, "%(title)s.%(ext)s")

        # yt-dlp options
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": output_path,
            "quiet": True,
        }

        # Download audio
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:

            info = ydl.extract_info(youtube_url, download=True)

            # Video title
            title = info["title"]

            # Downloaded file path
            file_path = ydl.prepare_filename(info)

        # Open audio file
        with open(file_path, "rb") as file:

            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(file_path), file.read()),
                model="whisper-large-v3",
                response_format="text",
            )

        return title, youtube_url, transcription.strip()

transcriptions = []

for youtube_url in YOUTUBE_VIDEOS:
    transcriptions.append(transcribe(youtube_url))

df = pd.DataFrame(transcriptions, columns=["title", "url", "text"])

df.to_csv("text.csv", index=False)

print(df.head())

MAX_TOKENS = 500

tokenizer = tiktoken.get_encoding("cl100k_base")

df = pd.read_csv("text.csv")

df["tokens"] = df["text"].apply(
    lambda x: len(tokenizer.encode(str(x)))
)

df.head()


def split_into_many(text, max_tokens):
    # Split the text into sentences
    sentences = text.split('. ')

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    # Add the last chunk to the list of chunks
    if chunk:
        chunks.append(". ".join(chunk) + ".")

    return chunks


data = []
for row in df.iterrows():
    title = row[1]["title"]
    url = row[1]["url"]
    text = row[1]["text"]
    tokens = row[1]["tokens"]

    if tokens <= MAX_TOKENS:
        data.append((title, url, text))
    else:
        for chunk in split_into_many(text, MAX_TOKENS):
            data.append((title, url, chunk))

df = pd.DataFrame(data, columns=["title", "url", "text"])
df["tokens"] = df.text.apply(lambda x: len(tokenizer.encode(x)))
df.to_csv("video_text.csv", index=False)

file = "video_text.csv"

loader = CSVLoader(file_path=file)
docs = loader.load()

embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

pc= Pinecone()

PINECONE_INDEX = "another-tube"
embedding_dimension = 384
if PINECONE_INDEX not in pc.list_indexes():
    pc.create_index(
        PINECONE_INDEX,
        dimension=embedding_dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )


index = pc.Index(PINECONE_INDEX)
index.describe_index_stats()


vectors = []

# Loop through documents
for doc in docs:

    # Create embedding
    embedding = embeddings.embed_query(doc.page_content)

    # Create unique ID
    vector_id = str(uuid4())

    # Store vector
    vectors.append(
        {
            "id": vector_id,
            "values": embedding,
            "metadata": {
                "text": doc.page_content
            }
        }
    )

# Upload vectors to Pinecone
index.upsert(vectors=vectors)

print("Documents uploaded successfully")

query = "What is langchain?"

query_embedding = embeddings.embed_query(query)

results = index.query(
    vector=query_embedding,
    top_k=3,
    include_metadata=True
)

print(results)

for match in results["matches"]:

    print(match["score"])

    print(match["metadata"]["text"])

    print("\n")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature = 0.0)

query = "What does langchain do?"

# Convert query to embedding
query_embedding = embeddings.embed_query(query)

# Search Pinecone
results = index.query(
    vector=query_embedding,
    top_k=3,
    include_metadata=True
)

# Extract retrieved text
contexts = []

for match in results["matches"]:

    contexts.append(
        match["metadata"]["text"]
    )

# Combine context
context_text = "\n\n".join(contexts)

# Prompt
prompt = PromptTemplate.from_template(
    template = "Answer the question based only on the context below.\nContext: {context_text}\n Question:{query}"
)
formatted_prompt = prompt.format(
    context_text=context_text,
    query=query
)
# Generate answer
response = llm.invoke(formatted_prompt)

# Print response
print(response.content)


