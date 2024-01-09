from pydantic import BaseModel, Field, ValidationError
from typing import Dict


class Search(BaseModel):
    query: str = Field(min_length=3)

    @classmethod
    def from_dict(cls, params: Dict):
        try:
            return cls(**params)
        except ValidationError as e:
            raise ValueError(f"Validation error: {e}")
