# Use a non-root user to install packages
# FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Upgrade pip
RUN python -m pip install --upgrade pip
RUN apt-get update
RUN apt-get install git -y

# Install DGL
RUN pip install  dgl -f https://data.dgl.ai/wheels/cu116/repo.html

RUN mkdir -p /workspace 
WORKDIR /workspace 

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN python -c "import torch; print(torch.__version__)"
RUN python -c "import dgl; print(dgl.__version__)"
RUN python -c "import torch; print(torch.cuda.is_available())"

