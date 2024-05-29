# from us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-cu121.2-1.py310
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /neuralhydro

COPY requirements.txt ./requirements.txt

# Install the package dependencies in the requirements file
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ .

CMD ["python", "/neuralhydro/task.py"]
