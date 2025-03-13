FROM node:20-alpine3.20

WORKDIR /app

COPY . .

RUN npm ci && npm run build

EXPOSE 3000

CMD [ "npm","run","start" ]

