from langchain_community.document_loaders import csv_loader

loader = csv_loader(file_path = "file path")

docs = loader.load()
#laods one row as a document, for n rows n no of docs
print(len(docs))

print(docs[0])

# more docuemt loaders = "https://docs.langchain.com/oss/python/integrations/document_loaders"