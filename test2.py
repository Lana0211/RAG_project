import json
from typing import List

from langchain_community.document_loaders import JSONLoader
from langchain_community.retrievers import ChatGPTPluginRetriever
from langchain_core.documents import Document

loader = JSONLoader(
    file_path="./data/patient_id_to_vector.json"
)
data = loader.load()

def write_json(path: str, documents: List[Document]) -> None:
    results = [{"text": doc.page_content} for doc in documents]
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


write_json("foo.json", data)

retriever = ChatGPTPluginRetriever(url="http://0.0.0.0:8000", bearer_token="foo")

responce = retriever.invoke("Patient's past medical history.")
print(responce)
