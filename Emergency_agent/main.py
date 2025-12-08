from tools import FaissVectorStore
from tools import load_all_documents, splitter_documents, embed_chunks

# data loading
docs = load_all_documents('./documents')

# data splitting
chunks = splitter_documents(docs, chunk_size=1000, chunk_overlap=200)

# data embedding
embeddings = embed_chunks(chunks)

# data vectorisation
vs = FaissVectorStore(persist_dir='BasesVectorielle')

# Ajout du text dans les metadonnees
metadatas = [c.metadata for c in chunks]

for i, meta in enumerate(metadatas):
    meta['text']= chunks[i].page_content
    meta['source'] = 'Documents'
vs.add_embeddings(embeddings, metadatas)
vs.save() # Enregistrement de la bases vectorielle
