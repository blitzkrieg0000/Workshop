FROM node:10
WORKDIR /usr/src/app
COPY package.json .
RUN npm install

#Imaj değişince container da değişir çünkü container katmanlı mimari kullanır.
#Daha fazla cache kullanılması için alt kısma yazıyoruz
COPY server.js .    


EXPOSE 8080
CMD ["node", "server.js"]