from langchain_text_splitters import CharacterTextSplitter, Language

text = " paste your python code here"

splitter = CharacterTextSplitter(
    Language = Language.PYTHON,
    chunk_size = 300,
    chunk_overlap = 0
)

chunks = splitter.split_text(text)

print(len(chunks))

print(chunks[0])
