#### Запустить локально: 

##### Сначала задать пути в окружении:

 PATH_TO_MODEL=model.pkl (set PATH_TO_MODEL=model.pkl for Windows) 
 PATH_TO_TRANSFORMER=transformer.pkl (set PATH_TO_TRANSFORMER=transformer.pkl for Windows) 
 
 ##### Потом запустить: 
 
 uvicorn app:app --host 127.0.0.1


#### Загрузить и запустить докер: 
docker pull novokolg/online_inference:v3

docker run -p 8000:8000 novokolg/online_inference:v3

#### Запуск скрипта: 
python make_request.py