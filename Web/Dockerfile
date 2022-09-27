FROM python:3.8-slim-buster
ENV PYTHONUNBUFFRED=1 
WORKDIR /Group06

#COPY . /Group06

COPY requirments.txt requirments.txt
RUN pip install -r requirments.txt

COPY . /Group06

EXPOSE 8888
CMD ["sh", "start-server.sh"]