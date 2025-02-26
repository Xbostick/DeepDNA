FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime


LABEL project="DeepDNA"

WORKDIR /usr/src/app

COPY DeepDNA_Pipeline.ipynb ./
CMD mkdir DeepDNA_sources
COPY DeepDNA_sources ./DeepDNA_sources
CMD mkdir BERT_cache

# Update and install necessary packages
RUN apt update && \
    apt upgrade -y && \
    apt install -y wget
    
    
# Install Python dependencies
COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

#torch
ENV CUDA=1
#BERT
#["HG chipseq", "HG kouzine", "MM chipseq", "MM kouzine"]
ENV BERT_MODEL="HG kouzine"
#wandb
 
# Expose Jupyter Notebook port
EXPOSE 8888

# Define entry point and default command
ENTRYPOINT ["python3", "-m"]
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
