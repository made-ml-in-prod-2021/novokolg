#### Запустить локально: 

##### Сначала задать пути в окружении:

 PATH_TO_MODEL=model.pkl (set PATH_TO_MODEL=model.pkl for Windows) 
 PATH_TO_TRANSFORMER=transformer.pkl (set PATH_TO_TRANSFORMER=transformer.pkl for Windows) 
 
 ##### Потом запустить: 
 
 uvicorn app:app --host 127.0.0.1

##### Собрать докер: 
docker build -t novokolg/online_inference:v1 . 

##### Загрузить и запустить докер: 
docker build -t novokolg/online_inference:v1 . 

docker run -p 80:80 novokolg/online_inference:v1

##### Запуск скрипта: 
python make_request.py

#### Docker path:

https://hub.docker.com/layers/novokolg/online_inference/v1/images/sha256:e9e1a9e22cb60327a0839563c686aa12cc78aa352509e9e0d5a3c5d0cc257b2e
