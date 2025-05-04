FROM nvcr.io/nvidia/pytorch:20.10-py3

FROM python:3.8

RUN pip install --upgrade pip

COPY requirements.txt /code/

RUN pip install -r /code/requirements.txt

COPY ./ /code/

WORKDIR /code

CMD ["python", "start_api.py", "--port", "8018", "--host", "0.0.0.0", "--kubeai_host", "http://localhost:1235/v1"]
