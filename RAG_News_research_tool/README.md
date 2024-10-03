
# News Research Tool 

This is a user-friendly news research tool designed for effortless information retrieval. Users can input article URLs and ask questions to receive relevant informations.

## Features

- Load URLs or upload text files containing URLs to fetch article content.
- Process article content through LangChain's UnstructuredURL Loader
- Construct an embedding vector using GoogleGenerativeAIEmbeddings and leverage FAISS, a powerful similarity search library, to enable swift and effective retrieval of relevant information
- Interact with the LLM's by inputting queries and receiving answers along with source URLs.
