# Hybrid Flow Vector (HFV) for Encrypted Traffic Classification

**Skripsi (Undergraduate Thesis) Repository**

**Author:** Hanif Nur Ilham Sanjaya  
**NPM:** 2206059692  
**Institution:** Universitas Indonesia

---

## 📖 Project Overview

The massive adoption of Virtual Private Networks (VPNs) has significantly improved user privacy, but it has also introduced considerable challenges for network management and security by obfuscating traffic content. Traditional analysis methods like Deep Packet Inspection (DPI) are ineffective against encryption, creating a critical need for techniques capable of classifying traffic without decrypting its payload.

While many methods can detect the presence of a VPN, fine-grained classification of specific applications or categories inside the encrypted tunnel remains a complex problem. This research proposes and validates a novel flow-based model, the **Hybrid Flow Vector (HFV)**, designed to tackle this challenge evaluated on the **ISCX 2016 Dataset**.

The HFV model is a multi-modal feature vector that effectively combines deep learning spatial features with statistical and behavioral flow metrics, demonstrating that deep learning features and statistical features are highly complementary for robust encrypted traffic classification.

## 🧠 The HFV Model Architecture

The Hybrid Flow Vector is composed of three distinct feature sets extracted from network flows:

1. **Alpha ($\alpha$) - Deep Learning Features:**
   - **Dimension:** 128-dim vector.
   - **Source:** Extracted using a 1D-Convolutional Neural Network (1D-CNN) processing the raw payload bytes of the first few packets in a flow.
   - **Purpose:** Captures complex, spatial, and hidden patterns within the encrypted payload sequence.

2. **Beta ($\beta$) / Delta ($\delta$) - Flow-Level Statistical Features:**
   - **Dimension:** 39-dim vector.
   - **Source:** Statistical metrics of the entire network flow (e.g., flow duration, total forward/backward packets, byte counts).
   - **Purpose:** Represents the overall macroscopic behavior and volume of the communication.

3. **Gamma ($\gamma$) - Burst-Level Statistical Features:**
   - **Dimension:** 37-dim vector.
   - **Source:** Statistical metrics detailing packet bursts (e.g., burst duration, bytes per burst).
   - **Purpose:** Captures the microscopic, temporal behavior and pacing of the application's communication.

*Note: The combined full model ($\alpha$ + $\beta$ + $\gamma$) achieved peak accuracies of 97.28% for binary classification (VPN vs. Non-VPN) and 81.88% for category classification. Inside VPN traffic specifically, the full hybrid model achieved an optimal 94.48% accuracy.*

## 📂 Repository Structure & Code Details

The codebase is built entirely using Python and Jupyter Notebooks, structured as an end-to-end experimental pipeline.

### 1. Data Preprocessing
*   **`final-flows_filter.ipynb`**
    *   This notebook is responsible for reading raw network capture files (`.pcap`), filtering out invalid or background noise traffic, and cleaning the flows to prepare valid datasets for feature extraction.

### 2. Feature Extraction Modules
*   **`alpha1.ipynb`** & **`alpha2.ipynb`**
    *   Handles the deep learning component. It extracts raw payloads from the cleaned flows, preprocesses them into fixed-length byte arrays, and utilizes the 1D-CNN to generate the 128-dimensional $\alpha$ feature vectors.
*   **`beta.ipynb`**
    *   Iterates through the PCAP flows to calculate and extract the 39-dimensional flow-level statistical metrics ($\beta$/$\delta$ features).
*   **`gamma.ipynb`**
    *   Focuses on the temporal dynamics, grouping packets into bursts and calculating the 37-dimensional burst-level statistical metrics ($\gamma$ features).

### 3. Data Integration
*   **`HFV_merge.ipynb`**
    *   This notebook acts as the joiner. It takes the independent feature vectors ($\alpha$, $\beta$, and $\gamma$) generated in the previous steps and aligns them per-flow to create the final unified Hybrid Flow Vector datasets (typically outputting as CSV or NumPy arrays).

### 4. Classification & Evaluation
*   **`HFV_classifier.ipynb`** / **`HFV_Experiment.ipynb`**
    *   The core machine learning experimental loop. It loads the merged HFV dataset and runs "tournaments" or ablation studies to evaluate various traditional classifiers (such as Random Forest).
    *   It evaluates different feature combinations (e.g., Alpha only, Beta only, Full Hybrid) across multiple classification tasks (Binary, Category, Application) and generates the evaluation metrics and plots.

### 5. LaTeX Thesis (`skripsi/`)
Contains the full LaTeX source code for the thesis document, split into chapters (`01-bab1.tex` through `05-bab5.tex`), the abstract, and conclusion.

### 6. Results & Figures (`img/`)
Contains generated plots and figures from the experiments, including:
*   Ablation study graphs (`ablation_study.png`)
*   Confusion matrices for various tasks (`confusion_matrix_...png`)
*   Feature importance (FI) charts (`fi_bar_...png`, `fi_pie_...png`)
*   Correlation matrices (`corr_matrix_...png`)
*   Hyperparameter tuning results.

## ⚙️ Environment Setup & Dependencies

To replicate the experiments, you need **Python 3** and **Jupyter**. The key libraries required are:

*   **`scapy`**: Essential for reading, parsing, and manipulating raw `.pcap` files.
*   **`numpy`** & **`pandas`**: For efficient data manipulation and tabular data structuring.
*   **`scikit-learn`**: Provides the machine learning classifiers (Random Forest, etc.) and evaluation metrics.
*   **`tensorflow`** / **`keras`**: The deep learning backend required for the 1D-CNN (Alpha feature extractor).
*   **`joblib`**: Used for parallel processing to speed up the heavy feature extraction phases.

## 🚀 How to Run the Pipeline

The notebooks are designed to be executed sequentially:

1.  **Preparation:** Place your raw ISCX 2016 PCAP files in the designated data directory.
2.  **Filter Flows:** Run `final-flows_filter.ipynb` to output cleaned flows.
3.  **Extract Features:** Run `alpha1.ipynb`/`alpha2.ipynb`, `beta.ipynb`, and `gamma.ipynb` independently to generate the three feature sets.
4.  **Merge Data:** Run `HFV_merge.ipynb` to combine the extracted features into a single dataset.
5.  **Train & Evaluate:** Run `HFV_Experiment.ipynb` (or `HFV_classifier.ipynb`) to train the models, perform the ablation study, and reproduce the classification metrics and plots.
