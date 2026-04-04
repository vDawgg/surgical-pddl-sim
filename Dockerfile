FROM nvcr.io/nvidia/isaac-sim:5.1.0

ENV DEBIAN_FRONTEND=noninteractive \
ACCEPT_EULA=Y \
PRIVACY_CONSENT=Y

USER root

WORKDIR /isaac-sim

COPY . /isaac-sim

RUN chown -R isaac-sim:isaac-sim /isaac-sim

USER isaac-sim

RUN ./python.sh -m pip install polars

ENTRYPOINT ["./python.sh", "main.py"]
