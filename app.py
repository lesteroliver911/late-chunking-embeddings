import logging
import numpy as np
import requests
from transformers import AutoModel, AutoTokenizer
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def chunk_by_sentences(input_text: str, tokenizer: AutoTokenizer) -> tuple:
    logger.info("Chunking text by sentences")
    inputs = tokenizer(input_text, return_tensors='pt', return_offsets_mapping=True)
    punctuation_mark_id = tokenizer.convert_tokens_to_ids('.')
    sep_id = tokenizer.sep_token_id
    token_offsets = inputs['offset_mapping'][0]
    token_ids = inputs['input_ids'][0]
    chunk_positions = [
        (i, int(start + 1))
        for i, (token_id, (start, end)) in enumerate(zip(token_ids, token_offsets))
        if token_id == punctuation_mark_id
        and (
            i + 1 < len(token_offsets) and (token_offsets[i + 1][0] - token_offsets[i][1] > 0
            or token_ids[i + 1] == sep_id)
        )
    ]
    chunks = [
        input_text[x[1] : y[1]]
        for x, y in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
    ]
    span_annotations = [
        (x[0], y[0]) for (x, y) in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
    ]
    logger.info(f"Created {len(chunks)} chunks")
    return chunks, span_annotations

def chunk_by_tokenizer_api(input_text: str) -> tuple:
    logger.info("Chunking text using Jina AI Tokenizer API")
    url = 'https://tokenize.jina.ai/'
    payload = {
        "content": input_text,
        "return_chunks": "true",
        "max_chunk_length": "1000"
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        response_data = response.json()
        chunks = response_data.get("chunks", [])
        chunk_positions = response_data.get("chunk_positions", [])
        span_annotations = [(start, end) for start, end in chunk_positions]
        logger.info(f"Created {len(chunks)} chunks using API")
        return chunks, span_annotations
    except requests.RequestException as e:
        logger.error(f"Error in API request: {e}")
        return [], []

def late_chunking(model_output: torch.Tensor, span_annotation: list, max_length: int = None) -> list:
    logger.info("Performing late chunking")
    token_embeddings = model_output[0]
    outputs = []
    for embeddings, annotations in zip(token_embeddings, span_annotation):
        if max_length is not None:
            annotations = [
                (start, min(end, max_length - 1))
                for (start, end) in annotations
                if start < (max_length - 1)
            ]
        pooled_embeddings = [
            embeddings[start:end].mean(dim=0)
            for start, end in annotations
            if (end - start) >= 1
        ]
        pooled_embeddings = [
            embedding.detach().cpu().numpy() for embedding in pooled_embeddings
        ]
        outputs.append(pooled_embeddings)
    logger.info(f"Created {len(outputs)} late-chunked embeddings")
    return outputs

def cos_sim(x: np.ndarray, y: np.ndarray) -> float:
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def main():
    logger.info("Starting Late Chunking demonstration")

    # Load model and tokenizer
    logger.info("Loading model and tokenizer")
    try:
        tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
        model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
    except Exception as e:
        logger.error(f"Error loading model or tokenizer: {e}")
        return

    # Input text
    input_text = "Berlin is the capital and largest city of Germany, both by area and by population. Its more than 3.85 million inhabitants make it the European Union's most populous city, as measured by population within city limits. The city is also one of the states of Germany, and is the third smallest state in the country in terms of area."
    logger.info(f"Input text: {input_text}")

    # Determine chunks
    chunks, span_annotations = chunk_by_sentences(input_text, tokenizer)
    logger.info(f"Chunks: {chunks}")

    # Traditional chunking
    logger.info("Performing traditional chunking")
    with torch.no_grad():
        embeddings_traditional_chunking = model.encode(chunks)

    # Late chunking
    logger.info("Performing late chunking")
    inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        model_output = model(**inputs)
    embeddings = late_chunking(model_output, [span_annotations])[0]

    # Compare similarities
    logger.info("Comparing similarities")
    with torch.no_grad():
        berlin_embedding = model.encode('Berlin')

    for chunk, new_embedding, trad_embedding in zip(chunks, embeddings, embeddings_traditional_chunking):
        sim_new = cos_sim(berlin_embedding, new_embedding)
        sim_trad = cos_sim(berlin_embedding, trad_embedding)
        logger.info(f'Chunk: "{chunk}"')
        logger.info(f'Similarity (new): {sim_new:.4f}')
        logger.info(f'Similarity (traditional): {sim_trad:.4f}')

    logger.info("Late Chunking demonstration completed")

if __name__ == "__main__":
    main()
