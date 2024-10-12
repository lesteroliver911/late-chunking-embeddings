
# Late Chunking Embeddings

This project demonstrates the late chunking technique and embedding comparison using transformers. The goal is to show how preserving contextual information across entire documents before chunking provides more precise retrieval and minimizes storage needs.

## Features

- Tokenizes and chunks input text by sentences.
- Implements both traditional and late chunking methods for embeddings.
- Compares cosine similarity of embeddings between chunks using both methods.
- Highlights the benefits of late chunking in improving retrieval precision.

## Sample Script

We provide a sample Python script that demonstrates the execution of late chunking with embeddings. This script allows you to test and experiment with both traditional and late chunking methods.

## Installation

To run the project, install the required dependencies:

```bash
pip install transformers torch requests numpy
```

## Usage

Run the main script:

```bash
python main.py
```

This script will perform traditional chunking, late chunking, and compare the similarities of embeddings with both methods.

## License

MIT License
