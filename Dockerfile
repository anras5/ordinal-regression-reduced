FROM python:3.11

WORKDIR /app

RUN apt-get update && apt-get install -y glpk-utils graphviz\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]