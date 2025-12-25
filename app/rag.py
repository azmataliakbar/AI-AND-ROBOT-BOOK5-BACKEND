# rag.py - ENHANCED VERSION with Better Out-of-Scope Handling

from typing import List, Optional
from sqlalchemy.orm import Session
import logging
from time import perf_counter

from .gemini_client import gemini_client
from .models import Chapter
from .database import SessionLocal
from .schemas import ChatResponse, Reference, SearchResult
from .logging_config import logger as structured_logger
from .vector_db import vector_db
from .embeddings import embedding_service

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self):
        self.embedding_service = embedding_service
        self.gemini_client = gemini_client
        self.vector_db = vector_db

    def get_response(
        self,
        query: str,
        user_id: Optional[str] = None,
        chapter_id: Optional[str] = None,
        threshold: float = 0.70,
        max_results: int = 5
    ) -> ChatResponse:
        """
        Enhanced RAG with quota-aware fallback and out-of-scope handling:
        1) Vector search (without filters to avoid Qdrant index errors)
        2) Enhanced context formatting
        3) Rich citation system
        4) Quota fallback: Shows book content when Gemini API quota exceeded
        5) Out-of-scope fallback: Helpful message when question not in book
        """
        start_time = perf_counter()

        try:
            structured_logger.info(
                "chat_request_received",
                user_id=user_id,
                query=query[:100] + "..." if len(query) > 100 else query,
                execution_time=None
            )

            # Step 1: Generate embedding
            query_embedding = self.embedding_service.embed_text(query)

            if query_embedding is None or (isinstance(query_embedding, list) and len(query_embedding) == 0):
                logger.warning(f"Embedding generation failed for user {user_id}, using Gemini fallback")
                gemini_response = self._safe_gemini(
                    self.gemini_client.generate_content, 
                    prompt=query
                )

                # Check if quota exceeded
                if gemini_response.get("quota_exceeded"):
                    fallback_message = self._build_out_of_scope_quota_message()
                    query_time_ms = int((perf_counter() - start_time) * 1000)

                    structured_logger.log_chat_interaction(
                        user_id=user_id or "anonymous",
                        query=query[:100] + "..." if len(query) > 100 else query,
                        response=fallback_message[:200],
                        confidence_score=0.0,
                        execution_time=query_time_ms,
                        source_type="embedding_failed_quota_exceeded",
                        chapter_id=chapter_id
                    )

                    return ChatResponse(
                        response=fallback_message,
                        confidence_score=0.0,
                        source_type="embedding_failed_quota_exceeded",
                        references=[],
                        query_time_ms=query_time_ms
                    )

                query_time_ms = int((perf_counter() - start_time) * 1000)

                structured_logger.log_chat_interaction(
                    user_id=user_id or "anonymous",
                    query=query[:100] + "..." if len(query) > 100 else query,
                    response=gemini_response["response"],
                    confidence_score=gemini_response["confidence"],
                    execution_time=query_time_ms,
                    source_type="fallback",
                    chapter_id=chapter_id
                )

                return ChatResponse(
                    response=gemini_response["response"],
                    confidence_score=gemini_response["confidence"],
                    source_type="fallback",
                    references=[],
                    query_time_ms=query_time_ms
                )

            # Step 2: Search WITHOUT filters (temporary fix to avoid Qdrant index errors)
            vector_results = self.vector_db.search_similar(
                query_embedding=query_embedding,
                threshold=threshold,
                limit=max_results,
                filters=None  # ✅ No filters to avoid index errors
            )

            if vector_results:
                # Build rich context with chapter info
                context = self._build_enhanced_context(vector_results)
                
                # Build detailed prompt
                enhanced_prompt = self._build_enhanced_prompt(query, context, vector_results)

                # Try Gemini with enhanced quota handling
                gemini_response = self._safe_gemini(
                    self.gemini_client.chat_with_context,
                    message=enhanced_prompt,
                    context=context
                )

                # ✅ CHECK FOR QUOTA EXCEEDED
                if gemini_response.get("quota_exceeded"):
                    # Return fallback with book content
                    fallback_response = self._build_quota_fallback_response(
                        query=query,
                        context=context,
                        vector_results=vector_results
                    )
                    
                    references = self._build_enhanced_references(vector_results)
                    query_time_ms = int((perf_counter() - start_time) * 1000)

                    structured_logger.log_chat_interaction(
                        user_id=user_id or "anonymous",
                        query=query[:100] + "..." if len(query) > 100 else query,
                        response=fallback_response[:200],
                        confidence_score=0.5,
                        execution_time=query_time_ms,
                        source_type="quota_fallback",
                        chapter_id=chapter_id
                    )

                    return ChatResponse(
                        response=fallback_response,
                        confidence_score=0.5,
                        source_type="quota_fallback",
                        references=references,
                        query_time_ms=query_time_ms
                    )

                # Build rich references with citations
                references = self._build_enhanced_references(vector_results)

                # Add citation markers to response
                enhanced_response = self._add_citations_to_response(
                    gemini_response["response"],
                    vector_results
                )

                query_time_ms = int((perf_counter() - start_time) * 1000)

                structured_logger.log_chat_interaction(
                    user_id=user_id or "anonymous",
                    query=query[:100] + "..." if len(query) > 100 else query,
                    response=enhanced_response[:200],
                    confidence_score=gemini_response["confidence"],
                    execution_time=query_time_ms,
                    source_type="qdrant",
                    chapter_id=chapter_id
                )

                return ChatResponse(
                    response=enhanced_response,
                    confidence_score=gemini_response["confidence"],
                    source_type="qdrant",
                    references=references,
                    query_time_ms=query_time_ms
                )
            else:
                # No vector results → Gemini fallback
                logger.info(f"No vector results found for query, using Gemini fallback for user {user_id}")

                gemini_response = self._safe_gemini(
                    self.gemini_client.generate_content, 
                    prompt=query
                )

                # ✅ CHECK IF QUOTA EXCEEDED (Out-of-scope + No quota)
                if gemini_response.get("quota_exceeded"):
                    # Custom message when both Qdrant and Gemini fail
                    fallback_message = self._build_out_of_scope_quota_message()

                    query_time_ms = int((perf_counter() - start_time) * 1000)

                    structured_logger.log_chat_interaction(
                        user_id=user_id or "anonymous",
                        query=query[:100] + "..." if len(query) > 100 else query,
                        response=fallback_message[:200],
                        confidence_score=0.0,
                        execution_time=query_time_ms,
                        source_type="out_of_scope_quota_exceeded",
                        chapter_id=chapter_id
                    )

                    return ChatResponse(
                        response=fallback_message,
                        confidence_score=0.0,
                        source_type="out_of_scope_quota_exceeded",
                        references=[],
                        query_time_ms=query_time_ms
                    )

                # Normal fallback response (when Gemini works but no book content)
                query_time_ms = int((perf_counter() - start_time) * 1000)

                structured_logger.log_chat_interaction(
                    user_id=user_id or "anonymous",
                    query=query[:100] + "..." if len(query) > 100 else query,
                    response=gemini_response["response"],
                    confidence_score=gemini_response["confidence"],
                    execution_time=query_time_ms,
                    source_type="fallback",
                    chapter_id=chapter_id
                )

                return ChatResponse(
                    response=gemini_response["response"],
                    confidence_score=gemini_response["confidence"],
                    source_type="fallback",
                    references=[],
                    query_time_ms=query_time_ms
                )

        except Exception as e:
            query_time_ms = int((perf_counter() - start_time) * 1000)

            structured_logger.log_error(
                error_type="RAG_SERVICE_ERROR",
                error_message=str(e),
                endpoint="/chat",
                user_id=user_id
            )

            return ChatResponse(
                response="I encountered an error while processing your request. Please try again.",
                confidence_score=0.0,
                source_type="error",
                references=[],
                query_time_ms=query_time_ms
            )

    def _build_out_of_scope_quota_message(self) -> str:
        """
        Build a helpful message when question is out-of-scope AND quota exceeded
        """
        return """**📚 Question Not Found in Textbook**

I couldn't find relevant content in the **Physical AI & Humanoid Robotics** textbook for your question.

**This chatbot specializes in:**
- 🤖 **ROS 2 Fundamentals** - Nodes, topics, services, actions
- 🎮 **Gazebo & Unity Simulation** - Physics engines, robot models, environments
- 🚀 **NVIDIA Isaac Platform** - Isaac Sim, Isaac Lab, GPU-accelerated robotics
- 👁️ **Vision-Language-Action Models** - VLA models, multimodal AI, natural language control

**Please ask questions related to:**
- Humanoid robotics concepts and implementation
- Robot simulation, control, and navigation
- ROS 2 programming and architecture
- Physical AI applications and frameworks
- Computer vision and sensor integration
- Motion planning and manipulation

**Examples of good questions:**
- "What is ROS 2 and how does it work?"
- "How do I simulate a humanoid robot in Gazebo?"
- "What are Vision-Language-Action models?"
- "How does TF2 handle coordinate transformations?"

*Note: The AI assistant is temporarily unavailable due to quota limits. When available, it can answer general questions, but for best results, please ask about topics covered in the textbook.*"""

    def _build_quota_fallback_response(
        self, 
        query: str, 
        context: str, 
        vector_results: List[dict]
    ) -> str:
        """
        Build a helpful fallback response when quota is exceeded
        Shows relevant book content directly
        """
        # Get chapter information
        chapters_info = []
        for result in vector_results[:3]:
            ch_num = result.get("chapter_number", "")
            ch_title = result.get("chapter_title", "")
            if ch_num and ch_title:
                chapters_info.append(f"Chapter {ch_num}: {ch_title}")
            elif ch_title:
                chapters_info.append(ch_title)
        
        chapters_str = ", ".join(chapters_info) if chapters_info else "relevant chapters"
        
        # Extract content preview (first 1000 chars of context)
        content_preview = context[:1000] if len(context) > 1000 else context
        
        fallback_response = f"""**[AI Summary Temporarily Unavailable - Quota Limit Reached]**

Your question: *"{query}"*

I found relevant information from **{chapters_str}**. Here's the content from your textbook:

---

{content_preview}

{'...' if len(context) > 1000 else ''}

---

**Note:** The AI model has reached its daily quota limit. The content above is directly from your Physical AI & Humanoid Robotics textbook. 

**To get a full answer:**
- Wait a few minutes and try again (quota resets periodically)
- Read the complete chapter sections referenced above
- The quota fully resets at midnight PST

📚 **Sources:** {chapters_str}"""
        
        return fallback_response

    def _build_enhanced_context(self, vector_results: List[dict]) -> str:
        """
        Build rich context with chapter structure
        """
        context_parts = []
        
        for i, result in enumerate(vector_results, 1):
            chapter_num = result.get("chapter_number", "")
            chapter_title = result.get("chapter_title", "Unknown Chapter")
            module = result.get("module", "")
            content = result.get("content", "")
            
            section_header = f"=== Chapter {chapter_num}: {chapter_title}"
            if module:
                section_header += f" ({module})"
            section_header += " ===\n"
            
            context_parts.append(section_header + content + "\n")
        
        return "\n".join(context_parts)

    def _build_enhanced_prompt(self, query: str, context: str, vector_results: List[dict]) -> str:
        """Build enhanced prompt with chapter awareness"""
        chapters_info = []
        for result in vector_results:
            ch_num = result.get("chapter_number", "")
            ch_title = result.get("chapter_title", "")
            if ch_num and ch_title:
                chapters_info.append(f"Chapter {ch_num}: {ch_title}")
            elif ch_title:
                chapters_info.append(ch_title)
        
        chapters_str = ", ".join(chapters_info[:3]) if chapters_info else "the book"
        
        prompt = f"""You are an AI assistant helping students learn Physical AI and Humanoid Robotics.

The user asked: "{query}"

I found relevant information from {chapters_str}.

Please provide a comprehensive answer based ONLY on the book content below. If the information isn't in the content, say so clearly.

When you reference specific information, mention which chapter it comes from (e.g., "In Chapter 1...").

Book Content:
{context}

Provide a detailed, educational response:"""
        
        return prompt

    def _build_enhanced_references(self, vector_results: List[dict]) -> List[Reference]:
        """Build rich reference objects"""
        references = []
        
        for result in vector_results:
            ref = Reference(
                chapter_id=result.get("chapter_id"),
                chapter_title=result.get("chapter_title"),
                section=result.get("section_title") or result.get("section_id"),
                chapter_number=result.get("chapter_number"),
                module=result.get("module"),
                similarity_score=result.get("similarity_score"),
                content_preview=result.get("content", "")[:150] + "..."
            )
            references.append(ref)
        
        return references

    def _add_citations_to_response(self, response: str, vector_results: List[dict]) -> str:
        """
        Add chapter citations at the end
        """
        if not vector_results:
            return response
        
        # Build citation footer
        chapters = []
        for result in vector_results[:3]:  # Top 3 sources
            ch_num = result.get("chapter_number", "")
            ch_title = result.get("chapter_title", "")
            if ch_title:
                if ch_num:
                    chapters.append(f"Chapter {ch_num}: {ch_title}")
                else:
                    chapters.append(ch_title)
        
        if chapters:
            citation_footer = f"\n\n📚 **Sources:** {', '.join(chapters)}"
            return response + citation_footer
        
        return response

    def get_search_results(
        self, 
        query: str, 
        threshold: float = 0.7, 
        max_results: int = 5
    ) -> List[SearchResult]:
        """Get enhanced search results"""
        try:
            query_embedding = self.embedding_service.embed_text(query)

            if query_embedding is None or (isinstance(query_embedding, list) and len(query_embedding) == 0):
                logger.warning("Embedding generation failed for search, returning empty results")
                return []

            # Search WITHOUT filters (temporary fix)
            vector_results = self.vector_db.search_similar(
                query_embedding=query_embedding,
                threshold=threshold,
                limit=max_results,
                filters=None  # ✅ No filters
            )

            return [
                SearchResult(
                    chapter_id=result.get("chapter_id"),
                    chapter_number=result.get("chapter_number", ""),
                    chapter_title=result.get("chapter_title"),
                    module=result.get("module", ""),
                    content_snippet=(
                        (result.get("content", "")[:200] + "...") 
                        if len(result.get("content", "")) > 200
                        else result.get("content", "")
                    ),
                    similarity_score=result.get("similarity_score"),
                    topics=result.get("topics", [])
                )
                for result in vector_results
            ]
        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            return []

    def _get_chapter_title(self, chapter_id: str) -> str:
        """Get chapter title from database"""
        db: Session = SessionLocal()
        try:
            if not chapter_id:
                return "Unknown Chapter"
            chapter = db.query(Chapter).filter(Chapter.id == chapter_id).first()
            return chapter.title if chapter else f"Chapter {chapter_id}"
        except Exception:
            return f"Chapter {chapter_id}"
        finally:
            db.close()

    def _safe_gemini(self, fn, **kwargs) -> dict:
        """
        Wrap Gemini calls for consistent response
        Now properly detects quota_exceeded flag
        """
        try:
            resp = fn(**kwargs)
            if not isinstance(resp, dict):
                raise ValueError("Gemini returned non-dict response")
            
            # Check for quota exceeded flag from the response
            if resp.get("quota_exceeded"):
                return {
                    "response": resp.get("response", "Quota exceeded"),
                    "confidence": resp.get("confidence", 0.5),
                    "safety_ratings": resp.get("safety_ratings", []),
                    "quota_exceeded": True
                }
            
            text = resp.get("response") or resp.get("text") or ""
            confidence = resp.get("confidence", 0.0)
            safety = resp.get("safety_ratings", [])
            
            return {
                "response": text, 
                "confidence": confidence, 
                "safety_ratings": safety,
                "quota_exceeded": False
            }
        except Exception as e:
            error_str = str(e).lower()
            
            # ✅ Detect quota errors from exception message
            if "quota" in error_str or "429" in error_str or "resourceexhausted" in error_str:
                logger.warning(f"Gemini quota exceeded detected in exception: {str(e)[:100]}")
                
                return {
                    "response": "[QUOTA_EXCEEDED]",
                    "confidence": 0.0,
                    "safety_ratings": [],
                    "quota_exceeded": True  # ✅ CRITICAL FLAG
                }
            
            logger.error(f"Gemini call failed (non-quota error): {e}")
            return {
                "response": "I encountered an error while processing your request. Please try again.",
                "confidence": 0.0,
                "safety_ratings": [],
                "quota_exceeded": False
            }


# Global instance
rag_service = RAGService()