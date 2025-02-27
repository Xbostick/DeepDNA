# DeepDNA: Generative and Predictive DNA Sequence Analysis

## Overview
DeepDNA is a deep learning-based research project focused on the generation and analysis of synthetic DNA sequences with biologically relevant properties. The study explores the application of generative adversarial networks (GANs) and regression models to analyze DNA sequence structures and predict key biological characteristics. This work is particularly relevant in the field of computational biology, where the generation of high-quality synthetic sequences can aid in understanding gene regulation and DNA sequence variation.

The repository includes:
- **DeepDNA_Pipeline.ipynb** – The main Jupyter Notebook containing model training and evaluation steps.
- **DeepDNA_sources/** – Source code and scripts required for the pipeline.
- **Dockerfile** – A ready-to-use containerized environment.
- **requirements.txt** – List of dependencies for running the project.
- **sample_data/** – Example DNA sequences used for training and evaluation.

## Research Background and Motivation
The ability to generate synthetic DNA sequences with specific biological properties has significant implications in genetics, bioengineering, and pharmaceutical development. Many genetic studies rely on experimental data, which can be expensive and time-consuming to obtain. In this project, we investigate the use of deep learning-based approaches to generate sequences that closely resemble naturally occurring DNA structures while preserving important functional properties. 

Our methodology builds upon previous studies in sequence generation and functional annotation. The main challenges include ensuring the biological validity of synthetic sequences and accurately predicting their potential impact on gene expression. We leverage state-of-the-art deep learning models, including ResNet-based GANs and a fine-tuned BERT model, to enhance the accuracy of our generative process.

## Methodology
### 1. DNA Sequence Generation
- We use a generative adversarial network (GAN) based on ResNet to produce synthetic DNA sequences.
- The generator is trained to create sequences that mimic real genomic data, while the discriminator assesses their authenticity.
- The Wasserstein loss function is employed to stabilize training and improve convergence【Arjovsky et al., 2017】.

### 2. Biological Property Prediction
- Regression models are implemented to predict sequence-related biological attributes, such as transcription initiation rates and chromatin accessibility.
- BERT-based sequence embeddings are used to validate the generated sequences and compare them with real genomic data【Umerenkov et al., 2023】.
- Classification models are employed to determine whether a generated sequence belongs to a particular functional category.

### 3. Model Training and Optimization
- The dataset consists of experimentally validated DNA sequences extracted from genomic databases.
- Sequences are preprocessed, normalized, and encoded for deep learning models.
- The training process involves iterative optimization to minimize loss while maintaining sequence diversity.

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/DeepDNA.git
cd DeepDNA
```

### 2. Using Docker (Recommended)
Ensure you have Docker installed. Then, build and run the container:
```bash
docker build -t deepdna .
docker run --gpus all -p 8888:8888 deepdna
```
This will start a Jupyter Notebook server. Access it by opening:
```
http://localhost:8888/
```

### 3. Manual Installation (Without Docker)
If you prefer to install dependencies manually:
```bash
python -m venv venv
source venv/bin/activate  # (On Windows: venv\Scripts\activate)
pip install --upgrade pip
pip install -r requirements.txt
jupyter notebook
```

## Usage
1. Open `DeepDNA_Pipeline.ipynb` in Jupyter Notebook.
2. Load the dataset or use provided sample data.
3. Train the model to generate DNA sequences.
4. Use BERT-based validation to assess sequence biological relevance.
5. Analyze results using built-in visualization tools.

## Experimental Results
### Key Findings
- **Generated sequences exhibit patterns consistent with real genomic DNA**, with similar nucleotide distributions and motif frequencies.
- **BERT-based validation confirms that synthetic sequences maintain biologically meaningful characteristics**, suggesting potential use in in-silico studies.
- **GAN-based approach outperforms classical statistical models in generating diverse sequences**, reducing mode collapse issues common in simpler generative frameworks.

### Figures
- **Model Architecture:** ![image](https://github.com/user-attachments/assets/313659c8-7552-4b2b-a1e5-4a7fd523f90c)


- **Sample Generated Sequences:** ![image](https://github.com/user-attachments/assets/77e7fdbc-8bf8-45a1-a548-7f5a2535ad28)


- **Training Performance:** ![image](https://github.com/user-attachments/assets/5705bd47-c975-4b51-9ce4-1e86ccdc69ad)


## References
This project is based on research in generative models and DNA sequence analysis. Key references include:
1. Zrimec J., et al. *Controlling gene expression with deep generative design of regulatory DNA* // Nat Commun, 2022.
2. Gressel S., et al. *CDK9-dependent RNA polymerase II pausing controls transcription initiation* // Elife, 2017.
3. He K., et al. *Deep residual learning for image recognition* // CVPR, 2016.
4. Arjovsky M., et al. *Wasserstein GAN* // stat. ML, 2017.
5. Bailey T. L., et al. *MEME SUITE: tools for motif discovery and searching* // Nucleic acids research, 2009.
6. Umerenkov D., et al. *Z-flipon variants reveal the many roles of Z-DNA and Z-RNA in health and disease* // Life Sci Alliance, 2023.

## Future Work
- **Enhancing Model Interpretability:** Exploring attention-based models to better understand DNA sequence representations.
- **Incorporating Epigenetic Features:** Expanding input features to include histone modifications and chromatin accessibility data.
- **Experimental Validation:** Collaborating with wet-lab researchers to validate synthetic sequences in biological assays.

## Contribution
Contributions are welcome! Feel free to fork the repository and submit a pull request.

## License
This project is licensed under the MIT License.
