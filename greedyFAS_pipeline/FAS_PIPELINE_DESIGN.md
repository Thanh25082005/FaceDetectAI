# 🧬 End-to-End Pipeline: Feature Architecture Similarity (FAS) using greedyFAS

## 📋 Overview
This pipeline automates the process of comparing protein feature architectures between a set of Seed proteins and Query proteins. It uses `greedyFAS` to calculate similarity scores based on domain architectures (Pfam, Coils, etc.).

## 🔄 Pipeline Flow

```mermaid
graph TD
    A[Input: seed.fa, query.fa] --> B[Setup Environment]
    B --> C{Tools Installed?}
    C -->|No| D[fas.setup (Install PFAM/COILS)]
    C -->|Yes| E[Skip Setup]
    D --> E
    E --> F[Annotation Stage]
    F --> G[fas.doAnno (Seed)]
    F --> H[fas.doAnno (Query)]
    G --> I[seed.json]
    H --> J[query.json]
    I --> K[FAS Calculation]
    J --> K
    K --> L[fas.run / greedyFAS]
    L --> M[Output: Scores & Architectures]
```

## 🛠️ Step-by-Step Design

### 1. Environment Setup & Installation
- **Goal**: Ensure `greedyFAS` and dependencies are ready.
- **Action**: Check for `greedyFAS` executable. If missing, clone/install.
- **Data**: Run `fas.setup` to download Pfam databases and configure `annoTools.txt`.
- **Constraint**: Skip restricted license tools (SignalP, TMHMM) if not available using `--omit-restricted` flag (if supported) or by editing `annoTools.txt`.

### 2. Protein Annotation
- **Goal**: Identify secondary structures and domains for all proteins.
- **Tool**: `fas.doAnno`
- **Logic**:
    - Check if annotation cache exists (to save time).
    - Run annotation for `seed.fa` -> `seed.anno.json`.
    - Run annotation for `query.fa` -> `query.anno.json`.

### 3. FAS Calculation
- **Goal**: Compare architectures.
- **Tool**: `fas.run` (or logic implementing greedyFAS algorithm).
- **Logic**:
    - Input: Annotated JSONs from previous step.
    - Process: Pairwise comparison (N x N or N x M).
    - Metric: Feature Architecture Similarity score.

### 4. Output Generation
- **Goal**: Human-readable and parsable results.
- **Files**:
    - `results_fas_scores.txt`: Matrix or list of scores.
    - `results_architectures.txt`: Visual representation of features.
    - `pipeline.log`: Execution logs.

## 📂 Directory Structure

```
project_root/
├── inputs/
│   ├── seed.fa
│   └── query.fa
├── outputs/
│   ├── annotations/
│   ├── results/
│   └── logs/
├── tools/
│   └── greedyFAS/
└── run_pipeline.sh
```
