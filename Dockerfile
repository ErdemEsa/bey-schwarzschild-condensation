FROM python:3.11-slim

LABEL maintainer="Erdem Esa <erdem.esa.71@gmail.com>"
LABEL project="BEY Schwarzschild Condensation"

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    gfortran \
    texlive-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-science \
    cm-super \
    dvipng \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace
ENV PYTHONPATH=/workspace
ENV MPLBACKEND=Agg
CMD ["python", "code/run_all_bey.py"]
