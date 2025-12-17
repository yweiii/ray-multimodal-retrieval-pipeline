# Ray Multimodal Retrieval Pipeline

A **practical, end‑to‑end multimodal embedding and retrieval system** built to demonstrate
hands‑on experience with **Ray Data, Actor Pools, and vector similarity search (FAISS)**.

The system ingests real images from LAION, computes CLIP embeddings using Ray,
stores embeddings as Parquet, builds a FAISS index, and supports **text → image retrieval**.
The scope is intentionally sized to run on a laptop while reflecting real production design patterns.

---

## Architecture & Design Paradigm

### High‑level Architecture

```
LAION Metadata (URLs + captions)
        │
        ▼
Image Download (img2dataset)
        │
        ▼
Raw Image Store
(local filesystem / object-store–like layout)
        │
        ▼
Manifest
(media_id = sha256(image_bytes))
        │
        ▼
Ray Data
  ├─ Forced partitioning (override_num_blocks)
  ├─ ActorPoolMapOperator
  └─ Batched CLIP embedding inference
        │
        ▼
Parquet Embeddings (Silver Layer)
model_version=clip_vit_b32_openai/
dt=YYYY-MM-DD/
        │
        ▼
FAISS Index (Gold Layer)
  ├─ index.faiss
  └─ mapping.parquet
        │
        ▼
Text → Image Retrieval
(cosine similarity)
```

---

## Core Design Paradigms

### 1. Actor‑based Distributed Inference (Ray)
- CLIP model is loaded **once per Ray actor**, not per task
- Uses `ActorPoolMapOperator` to amortize model load cost
- Batch inference balances throughput vs latency
- Parallelism is explicitly enforced to avoid single‑block execution

```python
ray.data.from_pandas(df, override_num_blocks=actors * 8)
```

This guarantees enough blocks to keep all actors busy and was verified via:
- Ray execution plans
- Ray dashboard (actors > 1, tasks > 1)

---

### 2. Idempotency & Deduplication
- Each image is assigned a stable natural key:
  ```
  media_id = sha256(image_bytes)
  ```
- Embedding jobs are safe to rerun
- FAISS index is built after deduplicating by `media_id`
- Prevents duplicate vectors from polluting retrieval results

---

### 3. Offline / Online Separation
- **Offline (Silver Layer)**  
  - Parquet embeddings
  - Versioned by `model_version` and `dt`
  - Replayable and auditable

- **Online (Gold Layer)**  
  - FAISS vector index
  - Optimized for low‑latency similarity search

This mirrors common production data platform patterns.

---

### 4. Vector Retrieval Strategy
- CLIP image and text embeddings are **L2‑normalized**
- FAISS `IndexFlatIP` is used
- Inner product on normalized vectors = **cosine similarity**

This provides exact nearest‑neighbor search suitable for datasets up to tens of thousands of items.

---

## Running the Pipeline

### 1. Compute Embeddings with Ray
```bash
python -m src.embed.run_embed_ray \
  --actors 4 \
  --batch_size 8 \
  --limit 5000
```

Key execution signals:
- `ActorPoolMapOperator[MapBatches(EmbedUDF)]`
- `Actors > 1`
- `Tasks > 1`

---

### 2. Build FAISS Index
```bash
python -m src.index.build_faiss
```

Artifacts:
- `index.faiss`
- `mapping.parquet`

---

### 3. Query (Text → Image)
```bash
python -m src.index.search_cli \
  --q "a dog running on grass" \
  --k 10
```

Returns top‑K images ranked by cosine similarity.

---

## Technologies Used
- **Ray Data** (Actor pools, batch processing)
- **OpenCLIP** (ViT‑B/32)
- **Parquet / PyArrow**
- **FAISS** (IndexFlatIP)
- **Python, NumPy, Pandas**

---

## What This Project Demonstrates
- Real Ray Data execution (not toy examples)
- Debugging and fixing single‑block parallelism issues
- Actor‑based batch inference design
- Vector similarity search fundamentals
- Clean separation of offline computation and online serving

This project focuses on **correct system design and execution semantics** rather than raw scale.