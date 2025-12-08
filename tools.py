import os
import faiss
import numpy as np
import pickle
import sqlite3
from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyMuPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from dotenv import load_dotenv
load_dotenv()


# Configuration des chemins
BASE_DIR = Path(__file__).resolve().parent
VECTOR_DB_DIR = BASE_DIR/'BasesVectorielle'
SQL_DB_PATH = BASE_DIR/'BaseHop.db'

# Variables de stockage des models
_EMBEDDING_MODEL = None
_VECTOR_STORE_INSTANCE = None

def embedding_model():
    '''Fonction pour charger le model d'embedding'''
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        print('[INFO] loading of embedding model ...')

        from sentence_transformers import SentenceTransformer
        _EMBEDDING_MODEL=SentenceTransformer('all-MiniLM-L6-v2')
    return _EMBEDDING_MODEL


# Fonction pour telecharger les documents
def load_all_documents(data_dir: str) -> List[Any]:
    """ Telechargement des documents """
    # Use project root data folder
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")
    
    documents = []

    # PDF files
    pdf_files = list(data_path.glob('**/*.pdf'))
    print(f"[DEBUG] Found {len(pdf_files)} PDF files: {[str(f) for f in pdf_files]}")
    for pdf_file in pdf_files:
        print(f"[DEBUG] Loading PDF: {pdf_file}")
        try:
            loader = PyMuPDFLoader(str(pdf_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} PDF docs from {pdf_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load PDF {pdf_file}: {e}")

    # CSV files
    csv_files = list(data_path.glob('**/*.csv'))
    print(f"[DEBUG] Found {len(csv_files)} CSV files: {[str(f) for f in csv_files]}")
    for csv_file in csv_files:
        print(f"[DEBUG] Loading CSV: {csv_file}")
        try:
            loader = CSVLoader(str(csv_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} CSV docs from {csv_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load CSV {csv_file}: {e}")


    print(f"[DEBUG] Total loaded documents: {len(documents)}")
    return documents

# Step 1 : documents splitting

def splitter_documents(docs, chunk_size=1000, chunk_overlap = 200):
    text_spliter= RecursiveCharacterTextSplitter(
    chunk_size = chunk_size,
    chunk_overlap = chunk_overlap,
    length_function = len,
    separators=['\n\n', '\n', ' ', ''],
    )

    split_docs = text_spliter.split_documents(docs)
    print(f'[INFO] {len(docs)} ont ete partitionner en {len(split_docs)} partition.')

    # show sample 
    if split_docs:
        print(f'Contenue du premier docs: {split_docs[0].page_content[0:200]}')
    
    return split_docs

def embed_chunks(chunks:List[Any])->np.ndarray:
    texts = [chunk.page_content for chunk in chunks]
    print(f'[INFO] Embedding for {len(texts)} chunks ...')

    from sentence_transformers import SentenceTransformer
    embeddings = SentenceTransformer("all-MiniLM-L6-v2").encode(texts)
    print(f'[INFO] Embedding shape : {embeddings.shape}')

    return embeddings




# Step 2 : Embedding 

class FaissVectorStore:
    def __init__(self, persist_dir: str = str(VECTOR_DB_DIR), dim: int = 384):
        self.persist_dir = persist_dir
        self.index = None
        self.metadata = []  # Stockera tout: text, source, page...
        self.dim = dim
        
        
    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any]):
        """Ajoute des vecteurs et leurs métadonnées"""
        if self.index is None:
            # Recherche Euclidienne
            self.index = faiss.IndexFlatL2(self.dim)
        
        # FAISS attend du float32
        if embeddings.dtype != 'float32':
            embeddings = embeddings.astype('float32')
            
        self.index.add(embeddings)
        self.metadata.extend(metadatas)
        print(f"[INFO] Added {embeddings.shape[0]} vectors to Faiss index.")

    def save(self):
        os.makedirs(self.persist_dir, exist_ok=True)
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        
        if self.index:
            faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] Saved Faiss index and metadata to {self.persist_dir}")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        
        if os.path.exists(faiss_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(faiss_path)
            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)
            return True
        return False

    def search(self, query_embedding: np.ndarray, top_k: int = 3):
        """Retourne les documents avec leurs scores"""
        if not self.index:
            return []
            
        # Recherche vecteur
        D, I = self.index.search(query_embedding.reshape(1, -1).astype('float32'), top_k)
        
        results = []
        # I[0] contient les indices, D[0] les distances
        for idx, dist in zip(I[0], D[0]):
            if idx < len(self.metadata) and idx >= 0:
                meta = self.metadata[idx]
                # Conversion Distance L2 -> Score de similarité (approx)
                # Score = 1 / (1 + distance)
                similarity = 1 / (1 + dist)
                results.append({
                    "content": meta.get("text", ""),
                    "metadata": meta,
                    "score": similarity
                })
        return results

# Fonction de chargement de la base vectorielle     
def get_vector_store():
    '''Chargement de la base vectorielle'''
    global _VECTOR_STORE_INSTANCE
    if _VECTOR_STORE_INSTANCE is None:
        print('[INIT] Chargement de la bases Faiss ...')
        vs = FaissVectorStore(persist_dir=str('BasesVectorielle'))
        if vs.load():
            _VECTOR_STORE_INSTANCE=vs
        else:
            print('[WARN] Base FAISS introuvable ou vide.')
    return _VECTOR_STORE_INSTANCE


# Fonction helper pour l'embedding
def embed_text(text:str)->np.ndarray:
    model = embedding_model()
    return model.encode([text])[0]

@tool
def consult_hospital_services(querry : str):
    """Fonction pour connaitre quel hopital est capable de traiter une pathologie specifique"""
    # print("[DEBUG] L'agent consulte la base des connaisses...")
    try:
        # Chargement de la donner :
        vs = get_vector_store()
        if not vs:
            return 'Error hospitals knowledge is empty'
        
        # Recherche du querry dans la base vectorielle 
        querry_emb = embed_text(querry)
        results = vs.search(querry_emb, top_k=3)

        # Formatage pour le LLM
        context = 'Information sur la capacite des hopitaux : \n'
        for res in results:
            context += f'{res['content']}\n'
        return context
    except Exception as e:
        return f'Erreur  outil services: {e}'


# Recherche des lits disponible 
@tool
def check_beds_availability():
    """Vérifie les lits disponibles par hôpital et par service."""
    # print("[DEBUG] L'agent consulte la base de données...")
    try:
        if not os.path.exists(SQL_DB_PATH):
            return 'Erreur : Base de données introuvable.'
        
        # Connexion 
        conn = sqlite3.connect(f"file:{SQL_DB_PATH}?mode=ro", uri=True)
        c = conn.cursor()
        
        # REQUÊTE SQL basée
        query = """
        SELECT h.nom, h.ville, s.nom_service, s.lits_disponibles 
        FROM service s
        JOIN hospital h ON s.hospital_id = h.id
        WHERE s.lits_disponibles > 0
        ORDER BY h.ville, h.nom
        """
        
        c.execute(query)
        results = c.fetchall()
        conn.close()

        if not results: 
            return "ALERTE: AUCUN LIT DISPONIBLE DANS LE RÉSEAU."
        
        txt = "DISPONIBILITÉS EN TEMPS RÉEL :\n"
        current_city = ""
        
        for nom_hopital, ville, service, lits in results:
            if ville != current_city:
                txt += f"\n SECTEUR {ville.upper()} :\n"
                current_city = ville
            txt += f"- {nom_hopital} : {service} ({lits} places)\n"
            
        return txt
        
    except Exception as e:
        return f"Erreur SQL : {e}"
    

    


