# Gemini Context: HFV Traffic Classification (Skripsi)

This `GEMINI.md` file provides context for the AI agent regarding the "HFV" (Hybrid Flow Vector) project, a final-year thesis (Skripsi) focused on encrypted VPN traffic classification.

## Project Overview

**Title:** Hybrid Flow Vector (HFV) for Encrypted Traffic Classification
**Goal:** To classify network traffic (specifically VPN vs. Non-VPN and specific applications) without decrypting the payload, using a novel multi-modal approach.
**Method:** The "HFV" model combines three feature sets:
1.  **Alpha ($\alpha$):** Deep learning features (128-dim) extracted via 1D-CNN from raw packet payloads.
2.  **Beta/Delta ($\beta$/$\delta$):** Flow-level statistical features (e.g., duration, packet counts).
3.  **Gamma ($\gamma$):** Burst-level statistical features.

## Directory Structure

### Root Directory (Codebase)
The root directory contains Jupyter Notebooks implementing the experimental pipeline.

*   **`final-flows_filter.ipynb`**: **Preprocessing.** Filters and cleans raw network flows (PCAP) to prepare valid datasets.
*   **`alpha1.ipynb` / `alpha2.ipynb`**: **Feature Extraction (Deep Learning).** Extracts raw payloads and likely trains/uses the 1D-CNN to generate the 128-dim $\alpha$ vector.
*   **`beta.ipynb`**: **Feature Extraction (Flow Stats).** Extracts the $\delta$ (Delta) features (statistical metrics of the entire flow).
*   **`gamma.ipynb`**: **Feature Extraction (Burst Stats).** Extracts the $\gamma$ (Gamma) features (statistical metrics of packet bursts).
*   **`HFV_merge.ipynb`**: **Data Integration.** Combines the individual feature vectors ($\alpha, \delta, \gamma$) into a single dataset.
*   **`HFV_classifier.ipynb`**: **Evaluation.** The main experimental loop. Runs "tournaments" to compare classifiers (e.g., Random Forest) on different feature combinations (Alpha only, Delta only, Full Hybrid, etc.).

### `skripsi/` (Thesis Document)
Contains the LaTeX source files for the thesis document.

*   `06-abstrak.tex`: Abstract (Indonesian/English).
*   `01-bab1.tex` - `05-bab5.tex`: Chapters 1 through 5 (Introduction, Literature Review, Methodology, Results, Conclusion).
*   `99-kesimpulan.tex`: Conclusion section.

## Development & Usage

### 1. Environment Setup
The project relies on **Python 3** and **Jupyter**.
**Key Dependencies:**
*   `scapy`: For reading and manipulating PCAP files.
*   `numpy`, `pandas`: For data manipulation.
*   `scikit-learn`: For machine learning classifiers (Random Forest, etc.).
*   `tensorflow` / `keras`: For the 1D-CNN model (Alpha component).
*   `joblib`: For parallel processing during feature extraction.

### 2. Execution Flow
The notebooks are designed to be run sequentially:
1.  **Preprocessing:** Run `final-flows_filter.ipynb` to prepare the PCAP data.
2.  **Extraction:** Run `alpha1.ipynb`, `beta.ipynb`, and `gamma.ipynb` to generate feature sets from the filtered PCAPs.
3.  **Merging:** Run `HFV_merge.ipynb` to join the features into a single dataset (CSV/NPY).
4.  **Classification:** Run `HFV_classifier.ipynb` to train models and evaluate performance.

### 3. Thesis Compilation
The thesis is written in LaTeX.
*   To compile, look for a main driver file (likely `main.tex` or similar, though not explicitly seen in the initial shallow list, it serves as the root for including the chapters).
*   Standard LaTeX build chain: `pdflatex` -> `bibtex` -> `pdflatex` -> `pdflatex`.

## Conventions
*   **Code:** Python in Jupyter Notebooks.
*   **Documentation:** Comments within notebooks explain specific steps (e.g., "Step 1: Extracting Delta Features").
*   **Language:** The thesis text is primarily in **Indonesian** (Bahasa Indonesia), as seen in filenames like `bab1` and content in `abstrak.tex`.
