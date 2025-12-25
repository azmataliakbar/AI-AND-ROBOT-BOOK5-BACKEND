# embeddings.py

from typing import List, Dict, Any, Optional
from .vector_db import vector_db
from .gemini_client import gemini_client
from .models import Chapter
from .database import SessionLocal
from sqlalchemy.orm import Session
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self):
        self.vector_db = vector_db
        self.gemini_client = gemini_client

    def create_chapter_embeddings(self, chapter: Chapter, force_rebuild: bool = False):
        """
        Create embeddings for a chapter by breaking it into sections and storing in vector DB
        """
        try:
            # If not force rebuild, check if embeddings already exist
            if not force_rebuild:
                existing_embeddings = self.vector_db.search_similar(
                    query_embedding=[0.1] * 768,  # Dummy embedding for checking
                    threshold=0.0,  # Any match
                    limit=1
                )
                # In a real implementation, we'd check specifically for this chapter
                # For now, we'll proceed with creation

            # Break chapter content into sections
            sections = self._split_content_into_sections(chapter.content, max_length=1000)

            chapter_embeddings = []
            section_ids = []
            contents = []

            for i, section in enumerate(sections):
                section_id = f"section_{i+1}"

                # Generate embedding for the section
                embedding = self.gemini_client.embed_content(section)

                if embedding:  # Only add if embedding was successful
                    chapter_embeddings.append(embedding)
                    section_ids.append(f"{chapter.id}_{section_id}")
                    contents.append(section)

            # Add all embeddings to vector database
            if chapter_embeddings:
                self.vector_db.add_batch_embeddings(
                    chapter_ids=[chapter.id] * len(chapter_embeddings),
                    section_ids=section_ids,
                    contents=contents,
                    embeddings=chapter_embeddings,
                    metadatas=[{"created_at": datetime.utcnow().isoformat()} for _ in chapter_embeddings]
                )

                logger.info(f"Created {len(chapter_embeddings)} embeddings for chapter {chapter.id}")
                return {
                    "chapter_id": chapter.id,
                    "sections_processed": len(chapter_embeddings),
                    "status": "success"
                }
            else:
                logger.warning(f"No embeddings created for chapter {chapter.id}")
                return {
                    "chapter_id": chapter.id,
                    "sections_processed": 0,
                    "status": "failed",
                    "message": "No embeddings could be generated"
                }

        except Exception as e:
            logger.error(f"Error creating embeddings for chapter {chapter.id}: {str(e)}")
            return {
                "chapter_id": chapter.id,
                "sections_processed": 0,
                "status": "error",
                "message": str(e)
            }

    def create_embeddings_for_chapters(self, chapter_ids: List[str], force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Create embeddings for multiple chapters
        """
        db: Session = SessionLocal()
        results = []
        processed_count = 0
        failed_count = 0

        try:
            for chapter_id in chapter_ids:
                # Get chapter from database
                chapter = db.query(Chapter).filter(Chapter.id == chapter_id).first()

                if not chapter:
                    results.append({
                        "chapter_id": chapter_id,
                        "status": "error",
                        "message": "Chapter not found"
                    })
                    failed_count += 1
                    continue

                # Create embeddings for this chapter
                result = self.create_chapter_embeddings(chapter, force_rebuild)
                results.append(result)

                if result["status"] == "success":
                    processed_count += 1
                else:
                    failed_count += 1

            status = "success" if failed_count == 0 else ("partial" if processed_count > 0 else "error")

            return {
                "processed_count": processed_count,
                "failed_count": failed_count,
                "total_requested": len(chapter_ids),
                "status": status,
                "details": results
            }

        finally:
            db.close()

    def search_similar_content(self, query: str, threshold: float = 0.7, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for content similar to the query
        """
        try:
            # Generate embedding for the query
            query_embedding = self.gemini_client.embed_content(query)

            if not query_embedding:
                logger.warning("Could not generate embedding for query")
                return []

            # Search in vector database
            results = self.vector_db.search_similar(
                query_embedding=query_embedding,
                threshold=threshold,
                limit=limit
            )

            return results

        except Exception as e:
            logger.error(f"Error searching for similar content: {str(e)}")
            return []

    def embed_text(self, text: str):
        """
        Generate embedding for a text string
        """
        try:
            embedding = self.gemini_client.embed_content(text)
            # Check if the embedding is valid (not None and not empty)
            if embedding is None or (isinstance(embedding, list) and len(embedding) == 0):
                return None
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding for text: {str(e)}")
            return None

    def _split_content_into_sections(self, content: str, max_length: int = 1000) -> List[str]:
        """
        Split content into sections of approximately max_length characters
        """
        sections = []
        paragraphs = content.split('\n\n')  # Split by double newlines (paragraphs)

        current_section = ""

        for paragraph in paragraphs:
            # If adding this paragraph would exceed max length
            if len(current_section) + len(paragraph) > max_length and current_section:
                # Save current section
                sections.append(current_section.strip())
                # Start new section with this paragraph
                current_section = paragraph
            else:
                # Add paragraph to current section
                if current_section:
                    current_section += "\n\n" + paragraph
                else:
                    current_section = paragraph

        # Add the last section if it exists
        if current_section:
            sections.append(current_section.strip())

        # If any section is still too long, split by sentences
        final_sections = []
        for section in sections:
            if len(section) <= max_length:
                final_sections.append(section)
            else:
                # Split long section by sentences
                sentences = section.split('. ')
                current_chunk = ""

                for sentence in sentences:
                    sentence_with_dot = sentence + ". "
                    if len(current_chunk) + len(sentence_with_dot) <= max_length:
                        current_chunk += sentence_with_dot
                    else:
                        if current_chunk:
                            final_sections.append(current_chunk.strip())
                        current_chunk = sentence_with_dot

                if current_chunk:
                    final_sections.append(current_chunk.strip())

        return final_sections


# Global instance
embedding_service = EmbeddingService()