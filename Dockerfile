FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y wget curl git openjdk-11-jdk && rm -rf /var/lib/apt/lists/*
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py310_23.11.0-2-Linux-x86_64.sh -O /tmp/mc.sh \
 && bash /tmp/mc.sh -b -p /opt/conda && rm /tmp/mc.sh
ENV PATH=/opt/conda/bin:$PATH
WORKDIR /workspace/EduGPT-RAG
COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -afy
SHELL ["conda", "run", "-n", "edugpt-rag", "/bin/bash", "-c"]
RUN python -m spacy download en_core_web_sm && python -c "import nltk; nltk.download('punkt')"
COPY . .
CMD ["conda", "run", "-n", "edugpt-rag", "bash", "run_all.sh"]
