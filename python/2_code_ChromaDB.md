# 1. Creating your own embedding function

## Code Format

### Link for above intent : ChromaDB
![Link of the code](https://cookbook.chromadb.dev/embeddings/bring-your-own-embeddings/#example-implementation)

### Code Format to be used
First you create a class that inherits from 'EmbeddingFunction[Documents]'. The Documents type is a list of Document objects. Each Document object has a text attribute that contains the text of the document. Chroma also supports multi-modal.

```
from chromadb.api.types import (
    Documents,
    EmbeddingFunction,
    Embeddings
)


class MyCustomEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(
            self,
            my_ef_param: str
    ):
        """Initialize the embedding function."""

    def __call__(self, input: Documents) -> Embeddings:
        """Embed the input documents."""
        return self._my_ef(input)
```

### Programmer's Interpretation:
The provided code snippet defines a `Protocol` class called `EmbeddingFunction` in Python. This class represents an interface for functions that take an input of type `D` and return an `Embeddings` object. Here are the key things to notice:

**1. `@runtime_checkable` Decorator:**
   - This decorator indicates that the `EmbeddingFunction` class is a runtime checkable protocol. This means that Python will dynamically check if an object implements the specified interface at runtime, rather than at compile time.

**2. `__call__` Method:**
   - This method is the core of the protocol and defines the expected behavior of any function that implements `EmbeddingFunction`. It takes an input of type `D` and returns an `Embeddings` object.
   - The `__init_subclass__` method ensures that `__call__` is defined for any subclass of `EmbeddingFunction`, raising an exception if it's not.
   - The `__call__` method also validates the returned `Embeddings` object using the `validate_embeddings` and `maybe_cast_one_to_many_embedding` functions.

**3. `embed_with_retries` Method:**
   - This method provides a convenient way to call the `__call__` method with retry logic. It takes an input of type `D` and optional retry keyword arguments.
   - It uses the `retry` function (likely from a third-party library) to retry the `__call__` method if it fails.

**4. Type Annotations:**
   - The code uses type annotations to specify the expected types of arguments and return values. This can improve code readability and maintainability.

**5. `cast` Function:**
   - The `cast` function is used to tell the type checker that the result of the `retry` function is an `Embeddings` object, even though the type checker might not be able to infer this automatically.

**In summary:**

- The `EmbeddingFunction` protocol defines a common interface for functions that embed data into a numerical representation.
- The `embed_with_retries` method provides a convenient way to call these functions with retry logic.
- Type annotations and the `cast` function are used to improve code readability and maintainability.

## Code Example

### Code

```
import importlib
from typing import Optional, cast

import numpy as np
import numpy.typing as npt
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings


class TransformerEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(
            self,
            model_name: str = "dbmdz/bert-base-turkish-cased",
            cache_dir: Optional[str] = None,
    ):
        try:
            from transformers import AutoModel, AutoTokenizer

            self._torch = importlib.import_module("torch")
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        except ImportError:
            raise ValueError(
                "The transformers and/or pytorch python package is not installed. Please install it with "
                "`pip install transformers` or `pip install torch`"
            )

    @staticmethod
    def _normalize(vector: npt.NDArray) -> npt.NDArray:
        """Normalizes a vector to unit length using L2 norm."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def __call__(self, input: Documents) -> Embeddings:
        inputs = self._tokenizer(
            input, padding=True, truncation=True, return_tensors="pt"
        )
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # mean pooling
        return [e.tolist() for e in self._normalize(embeddings)]
```

### Programmer's Interpretation

The provided code snippet defines a class called `TransformerEmbeddingFunction` that implements the `EmbeddingFunction` protocol. This class is designed to embed text documents using a transformer model, such as BERT. Here are the key things to notice:

**1. Inheritance:**
   - The class inherits from `EmbeddingFunction[Documents]`, indicating that it's a specific implementation of the `EmbeddingFunction` protocol for text documents.

**2. Initialization:**
   - The `__init__` method takes optional parameters for the model name and cache directory.
   - It imports the necessary modules from `transformers` and `torch` if they're available.
   - It creates instances of the tokenizer and model using the specified model name and cache directory.

**3. Normalization:**
   - The `_normalize` static method normalizes a vector to unit length using the L2 norm. This is a common preprocessing step for embeddings.

**4. `__call__` Method:**
   - This method is the core of the embedding function. It takes a list of documents as input and returns a list of embeddings.
   - It tokenizes the input documents using the tokenizer.
   - It passes the tokenized inputs to the transformer model and extracts the last hidden state.
   - It applies mean pooling to the last hidden state to obtain a single embedding for each document.
   - It normalizes the embeddings using the `_normalize` method.
   - It returns the embeddings as a list of lists.

**In summary:**

- The `TransformerEmbeddingFunction` class provides a convenient way to embed text documents using transformer models.
- It handles the initialization of the tokenizer and model, tokenization of the input documents, and normalization of the embeddings.
- It's a reusable component that can be used in various ChromaDB applications.

