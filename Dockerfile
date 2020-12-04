FROM python:3.5.7-slim

ENV PROGRAM_DIR=/app

RUN mkdir $PROGRAM_DIR
WORKDIR $PROGRAM_DIR

COPY requirements.txt /tmp

RUN pip install -U pip setuptools && \
  pip install -r /tmp/requirements.txt

COPY . $PROGRAM_DIR
RUN python setup.py install

CMD /bin/bash -c "python test.py"
