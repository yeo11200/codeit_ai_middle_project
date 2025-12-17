"""Vector similarity search module."""

from typing import List, Dict, Optional

from src.common.logger import get_logger


class VectorSearch:
    """Vector similarity search using ChromaDB."""
    
    def __init__(self, vector_store, embedder):
        """
        Initialize VectorSearch.
        
        Args:
            vector_store: VectorStore instance
            embedder: Embedder instance
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.logger = get_logger(__name__)
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query: Query text
            top_k: Number of results to return
            filters: Optional metadata filters (ChromaDB where clause format)
        
        Returns:
            List of search results with id, text, score, metadata
        """
        # Generate query embedding
        query_embedding = self.embedder.embed(query)
        
        # Search in ChromaDB
        collection = self.vector_store.collection
        
        # Build where clause if filters provided
        where_clause = filters if filters else None
        
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, 20),  # Max 20
                where=where_clause,
            )
            
            # Format results
            formatted_results = []
            
            if results["ids"] and len(results["ids"][0]) > 0:
                ids = results["ids"][0]
                documents = results["documents"][0]
                metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)
                distances = results["distances"][0] if results["distances"] else []
                
                for i, (doc_id, doc_text, metadata) in enumerate(zip(ids, documents, metadatas)):
                    # Convert distance to similarity score (1 - distance for cosine)
                    score = 1.0 - distances[i] if distances else 0.0
                    
                    formatted_results.append({
                        "id": doc_id,
                        "text": doc_text,
                        "score": score,
                        "metadata": metadata or {},
                    })
            
            return formatted_results
        
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            return []

