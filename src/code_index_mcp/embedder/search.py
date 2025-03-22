#!/usr/bin/env python3
import os
import pickle
import re
import logging
from typing import List, Dict, Any, Optional, Tuple, NamedTuple, Union
import numpy as np
from collections import defaultdict
from agents import Agent, Runner, function_tool

from .schema import Document, SearchResult, SearchMetadata

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SearchResult(NamedTuple):
    """Represents a search result with metadata."""
    document_id: str
    file_path: str
    file_name: str
    text_similarity: float
    code_similarity: float
    combined_score: float
    snippet: str
    metadata: Dict[str, Any]

def reciprocal_rank_fusion(
    text_scores: Dict[str, float], 
    code_scores: Dict[str, float], 
    k: int = 60
) -> List[Tuple[str, float]]:
    """
    Combine text and code rankings using Reciprocal Rank Fusion.
    
    Args:
        text_scores: Dictionary mapping document IDs to text similarity scores
        code_scores: Dictionary mapping document IDs to code similarity scores
        k: Constant to mitigate impact of high rankings (default: 60)
        
    Returns:
        List of (doc_id, score) tuples sorted in descending order by score
    """
    # Get all unique document IDs
    all_ids = set(text_scores.keys()) | set(code_scores.keys())
    
    # Create sorted lists for both types
    text_ranked = sorted([(doc_id, text_scores.get(doc_id, 0.0)) 
                          for doc_id in all_ids], 
                          key=lambda x: x[1], reverse=True)
    
    code_ranked = sorted([(doc_id, code_scores.get(doc_id, 0.0)) 
                          for doc_id in all_ids], 
                          key=lambda x: x[1], reverse=True)
    
    # Get ranks for each document (1-indexed)
    text_ranks = {doc_id: i+1 for i, (doc_id, _) in enumerate(text_ranked)}
    code_ranks = {doc_id: i+1 for i, (doc_id, _) in enumerate(code_ranked)}
    
    # Calculate RRF scores
    rrf_scores = {}
    
    for doc_id in all_ids:
        text_rank = text_ranks.get(doc_id, len(all_ids))
        code_rank = code_ranks.get(doc_id, len(all_ids))
        
        # RRF formula: 1 / (k + rank)
        text_rrf = 1.0 / (k + text_rank)
        code_rrf = 1.0 / (k + code_rank)
        
        # Sum RRF scores
        rrf_scores[doc_id] = text_rrf + code_rrf
    
    # Sort by RRF score (descending)
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_results

def get_document_snippet(document: Document, max_length: int = 200) -> str:
    """
    Extract a snippet from a document.
    
    Args:
        document: Document to extract snippet from
        max_length: Maximum length of the snippet
        
    Returns:
        Document snippet
    """
    content = document.content
    
    # For Python files, try to extract docstring
    if document.metadata.file_type == "py":
        # Look for triple-quoted docstring
        docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
        if docstring_match:
            docstring = docstring_match.group(1).strip()
            if docstring:
                return docstring[:max_length] + ("..." if len(docstring) > max_length else "")
    
    # If no docstring or not a Python file, return first part of content
    return content[:max_length] + ("..." if len(content) > max_length else "")

def group_by_field(
    results: List[SearchResult], 
    field_name: str, 
    top_k: int
) -> List[SearchResult]:
    """
    Group search results by a specified field.
    
    Args:
        results: List of search results
        field_name: Name of the field to group by
        top_k: Number of results to return after grouping
        
    Returns:
        Grouped search results
    """
    # Group results by the specified field
    groups = defaultdict(list)
    
    for result in results:
        # Get the group key
        if hasattr(result.metadata, field_name):
            key = getattr(result.metadata, field_name)
            groups[key].append(result)
        else:
            # If field doesn't exist, add to a default group
            groups["_default"].append(result)
    
    # Sort groups by highest score in each group
    sorted_groups = sorted(
        groups.items(), 
        key=lambda x: max(r.combined_score for r in x[1]) if x[1] else 0, 
        reverse=True
    )
    
    # Take top result from each group in round-robin fashion
    final_results = []
    group_indices = {group_key: 0 for group_key, _ in sorted_groups}
    
    while len(final_results) < top_k and any(
        group_indices[group_key] < len(group_items) 
        for group_key, group_items in sorted_groups
    ):
        for group_key, group_items in sorted_groups:
            if group_indices[group_key] < len(group_items):
                final_results.append(group_items[group_indices[group_key]])
                group_indices[group_key] += 1
                
                if len(final_results) >= top_k:
                    break
    
    # Sort the final results by score
    final_results.sort(key=lambda x: x.combined_score, reverse=True)
    
    return final_results[:top_k]

def search_embeddings(
    documents: List[Document],
    text_embeddings: Optional[np.ndarray] = None,
    code_embeddings: Optional[np.ndarray] = None,
    text_query_embedding: Optional[np.ndarray] = None,
    code_query_embedding: Optional[np.ndarray] = None,
    top_k: int = 10,
    group_by: Optional[str] = None,
) -> List[SearchResult]:
    """
    Search for documents similar to a query.
    
    Args:
        documents: List of documents
        text_embeddings: Text embeddings for documents
        code_embeddings: Code embeddings for documents
        text_query_embedding: Text embedding for query
        code_query_embedding: Code embedding for query
        top_k: Number of top results to return
        group_by: Field to group results by (e.g., 'file_path')
        
    Returns:
        List of search results
    """
    if not documents:
        return []
    
    # Calculate text similarities
    text_similarities = {}
    if text_embeddings is not None and text_query_embedding is not None:
        # Normalize embeddings
        text_embeddings_norm = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
        text_query_embedding_norm = text_query_embedding / np.linalg.norm(text_query_embedding)
        
        # Calculate cosine similarities
        text_scores = np.dot(text_embeddings_norm, text_query_embedding_norm)
        
        # Map to document IDs
        text_similarities = {documents[i].id: float(text_scores[i]) for i in range(len(documents))}
    
    # Calculate code similarities
    code_similarities = {}
    if code_embeddings is not None and code_query_embedding is not None:
        # Normalize embeddings
        code_embeddings_norm = code_embeddings / np.linalg.norm(code_embeddings, axis=1, keepdims=True)
        code_query_embedding_norm = code_query_embedding / np.linalg.norm(code_query_embedding)
        
        # Calculate cosine similarities
        code_scores = np.dot(code_embeddings_norm, code_query_embedding_norm)
        
        # Map to document IDs
        code_similarities = {documents[i].id: float(code_scores[i]) for i in range(len(documents))}
    
    # Combine scores using reciprocal rank fusion
    if text_similarities and code_similarities:
        combined_scores = reciprocal_rank_fusion(text_similarities, code_similarities)
    elif text_similarities:
        combined_scores = sorted(text_similarities.items(), key=lambda x: x[1], reverse=True)
    elif code_similarities:
        combined_scores = sorted(code_similarities.items(), key=lambda x: x[1], reverse=True)
    else:
        return []
    
    # Create search results
    doc_map = {doc.id: doc for doc in documents}
    search_results = []
    
    for doc_id, combined_score in combined_scores:
        if doc_id not in doc_map:
            continue
        
        doc = doc_map[doc_id]
        text_score = text_similarities.get(doc_id, 0.0)
        code_score = code_similarities.get(doc_id, 0.0)
        
        search_results.append(SearchResult(
            document_id=doc_id,
            file_path=doc.file_path,
            file_name=doc.file_name,
            text_similarity=text_score,
            code_similarity=code_score,
            combined_score=combined_score,
            snippet=get_document_snippet(doc),
            metadata=doc.metadata
        ))
    
    # Group results if requested
    if group_by is not None and hasattr(SearchMetadata, group_by):
        search_results = group_by_field(search_results, group_by, top_k)
    else:
        # Limit to top_k
        search_results = search_results[:top_k]
    
    return search_results

@function_tool
def search_embeddings_file(
    query: str, 
    embeddings_file: str, 
    top_k: int = 5,
    use_text_embeddings: bool = True,
    use_code_embeddings: bool = True,
    group_by: Optional[str] = None
) -> List[Dict]:
    """
    Search embeddings for documents similar to query.
    
    Args:
        query: Search query
        embeddings_file: Path to embeddings file
        top_k: Number of results to return
        use_text_embeddings: Whether to use text embeddings
        use_code_embeddings: Whether to use code embeddings
        group_by: Field to group results by (e.g., "module_name")
    """
    print(f"Searching for: {query}")
    print(f"Embeddings file: {embeddings_file}")
    print(f"Top k: {top_k}")
    
    # Check if file exists
    if not os.path.exists(embeddings_file):
        return [{"error": f"Embeddings file {embeddings_file} not found"}]
    
    # Load embeddings
    with open(embeddings_file, 'rb') as f:
        data = pickle.load(f)
    
    documents = data["documents"]
    
    # Check if we have dual embeddings or just single embeddings
    has_dual_embeddings = "text_embeddings" in data and "code_embeddings" in data
    
    if has_dual_embeddings:
        # Get embeddings
        text_embeddings = data["text_embeddings"]
        code_embeddings = data["code_embeddings"]
        
        # Get metadata
        metadata = data.get("metadata", {})
        text_model_name = metadata.get("text_model", "sentence-transformers/all-MiniLM-L6-v2")
        code_model_name = metadata.get("code_model", "microsoft/codebert-base")
        
        # Import model utilities
        from embedder.embedding_models import get_embedding_models
        
        # Get models
        embedding_model = get_embedding_models(text_model_name, code_model_name)
        
        # Generate query embeddings
        query_text_embedding, query_code_embedding = embedding_model.embed_query(query)
        
        # Search using dual embeddings
        results = search_embeddings(
            documents=documents,
            text_embeddings=text_embeddings,
            code_embeddings=code_embeddings,
            text_query_embedding=query_text_embedding,
            code_query_embedding=query_code_embedding,
            top_k=top_k,
            group_by=group_by
        )
    else:
        # Handle legacy single embeddings
        embeddings = np.array(data.get("embeddings", []))
        model_name = data.get("model", "all-MiniLM-L6-v2")
        
        # Load model
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        
        # Generate query embedding
        query_embedding = model.encode(query)
        
        # Calculate similarities
        similarities = np.dot(embeddings, query_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top results
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Convert to SearchResult format
        results = []
        for idx in top_indices:
            doc = documents[idx]
            similarity = float(similarities[idx])
            
            results.append(SearchResult(
                document_id=doc.id,
                file_path=doc.file_path,
                file_name=doc.file_name,
                text_similarity=similarity,
                code_similarity=0.0,
                combined_score=similarity,
                snippet=get_document_snippet(doc),
                metadata=doc.metadata
            ))
    
    # Format results for return
    formatted_results = []
    for result in results:
        result_dict = {
            "file_path": result.file_path,
            "file_name": result.file_name,
            "text_similarity": result.text_similarity,
            "code_similarity": result.code_similarity,
            "combined_score": result.combined_score,
            "snippet": result.snippet
        }
        
        # Add metadata
        for key, value in result.metadata.items():
            result_dict[key] = value
        
        # Add group information if available
        if group_by and hasattr(result.metadata, group_by):
            result_dict["group"] = getattr(result.metadata, group_by)
        
        formatted_results.append(result_dict)
        
        print(f"Match: {result.file_path} (Score: {result.combined_score:.4f})")
    
    return formatted_results

# Create search agent
search_agent = Agent(
    name="Search Agent",
    instructions="""
    You are a search agent for document embeddings. 
    
    Given a query and embeddings file:
    1. Search for documents similar to the query
    2. Return the top matches with relevant snippets
    3. Format the results clearly for the user
    
    The system supports dual embeddings (text and code) for more accurate search.
    You can choose to use only text embeddings, only code embeddings, or both.
    You can also group results by fields like module_name to get more diverse results.
    
    Respond in a clear and helpful manner, highlighting the most relevant information.
    """,
    tools=[search_embeddings_file]
) 