FROM nvcr.io/nvidia/pytorch:22.11-py3

RUN rm -rf /workspace/*
WORKDIR /workspace/unet

# Menasche add ------------
# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo 
#### Keeping this in is cargo-culty because sudo isnt' by default installed. so `$ sudo $command` gives a command not found error despite the fact that the user is in the sudo group. Neverthelss I will keep it around because I like the inline template. On top of all this, feels unsafe to gice container user sudo. 
### RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
# -------------------------

### menasche: we are adding this without testing it. We could rebuild the container and test but that would be annoying
### if something breaks RETURN HERE to fix things
ENV DISPLAY=:0
RUN apt-get update && apt-get install -y xorg xaut
#### ----------------------

ADD requirements.txt .
RUN pip install --user --no-cache-dir --upgrade --pre pip
RUN pip install --user --no-cache-dir -r requirements.txt

RUN pip install --user --no-cache-dir kornia
RUN pip install --user --no-cache-dir kornia-rs

#RUN pip install --no-cache-dir --upgrade --pre pip
#RUN pip install --no-cache-dir -r requirements.txt

ADD . .
