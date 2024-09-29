# Use the correct base image
FROM python:3.11.5

# Set the working directory
WORKDIR /project

RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"


# Clone the GIT repository
RUN git clone https://github.com/pethai2004/TTSC .

# Upgrade pip and setuptools
RUN pip3 install --upgrade pip setuptools
# copy git repository to the container
COPY . .
# Upgrade pip
RUN pip3 install --upgrade pip

# Create python virtual environment
RUN python3 -m venv .venv

# Activate virtual environment and install requirements
#RUN .venv/bin/pip3 install -r requirements.txt
RUN .venv/bin/pip3 install TTS --no-cache-dir
