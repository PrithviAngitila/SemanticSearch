from pydantic import BaseModel, Field, ValidationError
from typing import Dict


class LLMSearch(BaseModel):
    query: str = Field(min_length=3)

    @classmethod
    def from_dict(cls, params: Dict):
        try:
            return cls(**params)
        except ValidationError as e:
            raise ValueError(f"Validation error: {e}")
        

class Search(BaseModel):
    query: str = Field(min_length=3)
    search_type: str = 'bm25'

    @classmethod
    def from_dict(cls, params: Dict):
        try:
            return cls(**params)
        except ValidationError as e:
            raise ValueError(f"Validation error: {e}")
