from typing import List, Optional, Dict, Any

try:
    from pydantic import BaseModel, Field  # type: ignore
except ImportError:
    class BaseModel:  # type: ignore
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def dict(self) -> Dict[str, Any]:
            res = {}
            for k, v in self.__dict__.items():
                if hasattr(v, 'dict'):
                    res[k] = v.dict()
                elif isinstance(v, list):
                    res[k] = [i.dict() if hasattr(i, 'dict') else (i.value if hasattr(i, 'value') else i) for i in v]
                else:
                    res[k] = v
            return res

    def Field(default=..., **kwargs):
        return default

class ReIDMatchResult(BaseModel):
    """
    Strongly-typed result of a single Re-ID identity matching attempt.
    """
    candidate_track_id: Optional[int] = Field(None, description="Newly assigned candidate track ID")
    matched_track_id: Optional[int] = Field(None, description="Reconnected historical gallery track ID")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity score [0.0 to 1.0]")
    embedding_age: int = Field(..., ge=0, description="Age of matched gallery embedding in frames")
    reidentified: bool = Field(..., description="True if similarity score >= REID_SIMILARITY_THRESHOLD")

class ReIDFrameResult(BaseModel):
    """
    Per-frame output container from ReIdentificationService.
    """
    frame_index: int
    embeddings_extracted_count: int
    successful_matches_count: int
    failed_matches_count: int
    highest_similarity_score: float
    reid_latency_ms: float
    match_results: List[ReIDMatchResult]
