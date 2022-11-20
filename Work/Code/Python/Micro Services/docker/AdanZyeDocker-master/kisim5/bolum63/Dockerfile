FROM nginx:latest
LABEL author="Burakhan" gmail="abc@gmail.com"
ENV KULLANICI="BURAKHAN"
ARG RENK
RUN apt-get update &&\
    apt-get install apt-utils curl htop wget -y

WORKDIR /gecici

ADD https://wordpress.org/latest.tar.gz .

WORKDIR /usr/share/nginx/html

COPY html/${RENK}/ .

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 CMD curl -f http://127.0.0.1/ || exit 1

CMD [ "./script.sh" ]