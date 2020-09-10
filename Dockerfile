FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-4

LABEL maintainer = "Blake Harrison <blake@keruki.com>"

RUN mkdir /src

RUN chmod -R u+x /src/

ADD src /src/
WORKDIR /src

RUN apt-get clean
RUN apt-get update -y
RUN apt-get install -y mpich libmpich-dev
RUN rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENTRYPOINT ["python"]
CMD ["main.py"]
R