FROM python:3.10.0


WORKDIR /opt/ecb_crc_web
COPY . /opt/ecb_crc_web

# Install Chromium dependencies
#RUN apt-get update

RUN pip install -r requirements.txt


#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8211"]