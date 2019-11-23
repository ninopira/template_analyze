FROM hoge
MAINTAINER hoge

#  install jupyter dependencies
RUN apt-get update && apt-get install -y \
    git \
    less \
    autoconf \
    tmux \
    vim \
    htop \
    emacs \
    python-tk \
    man \
    wget \
    cmake \
    byobu \
    language-pack-ja \
    unzip \
    cmake \
    libgtk2.0-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    python3-numpy \
    python-tk \
    nscd \
    graphviz \
    python3-pip \
    protobuf-compiler \
    python-pil \
    python-lxml \
    && apt-get -y clean all \
    && rm -rf /var/lib/apt/lists/*

# set locale and timezone
ENV LANG="ja_JP.UTF-8"
ENV TZ="Asia/Tokyo"
RUN update-locale LANG=$LANG \
    && echo "${TZ}" > /etc/timezone \
    && rm /etc/localtime \
    && ln -s /usr/share/zoneinfo/$TZ /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata

RUN pip3 install -U pip

# settings for keras
ENV KERAS_BACKEND=tensorflow

# install other packages
RUN pip3 install jupyter
RUN pip3 install https://github.com/ipython-contrib/jupyter_contrib_nbextensions/tarball/master \
     && jupyter contrib nbextension install --user \
     && jupyter nbextension enable collapsible_headings/main

COPY requirements/*.txt /tmp/
RUN pip3 install -r /tmp/dev.txt

# for Keras.js
# EXPOSE 3000

# for Tensorboard
EXPOSE 6006

# for jupyterlab
EXPOSE 8888


CMD ["/usr/sbin/sshd", "-D"]
