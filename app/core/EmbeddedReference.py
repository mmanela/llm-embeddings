import hashlib
import uuid

NAMESPACE_UUID = uuid.UUID(int=1985)


def hasher(input_string: str) -> uuid.UUID:
    """Hash a string and returns the corresponding UUID."""
    hash_value = hashlib.sha1(input_string.encode("utf-8")).hexdigest()
    return uuid.uuid5(NAMESPACE_UUID, hash_value)


class EmbeddedReference(object):

    def __init__(self, content: str, metadata, embedding: list[float]):
        self.content = content
        self.metadata = metadata
        self.embedding = embedding
        self.hash_code = hasher(content)

    def forJson(self):
        return {
            'hash_code': str(self.hash_code),
            'content': self.content,
            'metadata': self.metadata
        }
