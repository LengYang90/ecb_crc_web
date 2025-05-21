FROM python:3.9-slim


WORKDIR /opt/ecb_crc_web
COPY . .

# Install Chromium dependencies
RUN apt-get update && apt-get upgrade -y

RUN pip install -r requirements.txt


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8211"]