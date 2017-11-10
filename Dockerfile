FROM tensorflow/tensorflow:1.2.0-devel-py3

# Copy pip requirements file to container.
ADD requirements.txt /tmp/

RUN apt update && apt install -y libsm6 # libxext6

# Install requirements using pip3.
RUN pip3 install -r /tmp/requirements.txt