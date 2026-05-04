import math
from src.embeddings.base import Vector, Vectors


def normalize_vector(vector: Vector) -> Vector:
    """Return a unit-length copy of the vector."""
    if not vector:
        raise ValueError("Cannot normalize an empty vector.")

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        raise ValueError("Cannot normalize a zero vector.")

    return [value / norm for value in vector]


def normalize_vectors(vectors: Vectors) -> Vectors:
    """Return unit-length copies for all vectors in the batch."""
    return [normalize_vector(vector) for vector in vectors]
