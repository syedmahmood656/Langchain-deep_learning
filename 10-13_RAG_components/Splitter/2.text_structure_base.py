 #recursive character text splitter

from langchain_text_splitters import CharacterTextSplitter

text = "paste your text"

splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0
)

chunk = splitter.split_text(text)

print(chunk)
print(len(chunk))