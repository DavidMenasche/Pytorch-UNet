FROM nvcr.io/nvidia/pytorch:22.11-py3

RUN rm -rf /workspace/*
WORKDIR /workspace/unet

# DMenasche add ------------
# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo 
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
# --------------------------

ADD requirements.txt .
RUN pip install --user --no-cache-dir --upgrade --pre pip
RUN pip install --user --no-cache-dir -r requirements.txt

#RUN pip install --no-cache-dir --upgrade --pre pip
#RUN pip install --no-cache-dir -r requirements.txt

ADD . .
