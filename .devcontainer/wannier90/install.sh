

# install gcc, required to compile wannier 90
apt-get update

apt-get install -y \
    gcc build-essential \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libopenmpi-dev \
    wget

# install wannier 90
WANNIER_VERSION="${VERSION:-"3.1.0"}"


cd /tmp/ && \
    wget  https://github.com/wannier-developers/wannier90/archive/v${WANNIER_VERSION}.tar.gz && \
    tar xzf v${WANNIER_VERSION}.tar.gz

cd /tmp/wannier90-${WANNIER_VERSION} && \
    cp ./config/make.inc.gfort ./make.inc && \
    make install