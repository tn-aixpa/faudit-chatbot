FROM python:3.10

RUN pip install --upgrade pip

COPY requirements.txt /code/

RUN pip install -r /code/requirements.txt

COPY ./ /code/

WORKDIR /code

RUN useradd -m -u 8877 nonroot
RUN chown -R 8877:8877 /code
USER 8877

ENTRYPOINT [ "python", "start_api.py" ]
