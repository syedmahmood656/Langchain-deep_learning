from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path="Folder path",
    glob='*.pdf', # as ther are only pdfs in our folder
    loader_cls=PyPDFLoader
)

docs = loader.lazy_load()

print(len(docs))

for documents in docs:
    print(documents.metadata)