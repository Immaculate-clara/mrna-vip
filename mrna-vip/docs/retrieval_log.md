# Retrieval log

Embedder: `hashing-bow-1024` - vector store: `numpy` - k = 3

| # | Question | Top result (first 90 chars) | Expected phrase in top-1 / top-3 | Latency (s) |
|---|----------|-----------------------------|----------------------------------|-------------|
| 1 | What are the storage requirements below -20C? | administration requires pediatric-formulated product and is not interchangeable with adult... | `ultra-low`: NO / yes | 0.0010 |
| 2 | What dose is considered high reactogenicity? | administration requires pediatric-formulated product and is not interchangeable with adult... | `100 micrograms`: yes / yes | 0.0009 |
| 3 | Which patients aged 65 or older need secondary clinician review? | administration requires pediatric-formulated product and is not interchangeable with adult... | `65 and older`: yes / yes | 0.0008 |
| 4 | How soon must a moderate or severe adverse event be documented? | administration requires pediatric-formulated product and is not interchangeable with adult... | `24 hours`: yes / yes | 0.0007 |
| 5 | Is anaphylaxis to a prior dose a contraindication? | administration requires pediatric-formulated product and is not interchangeable with adult... | `anaphylaxis`: yes / yes | 0.0007 |
| 6 | What should be done if the inflammatory biomarker is elevated at screening? | administration requires pediatric-formulated product and is not interchangeable with adult... | `biomarker`: yes / yes | 0.0006 |
