####Запустить локально:
#####Сначала задать пути в окружении
PATH_TO_MODEL=model.pkl (set PATH_TO_MODEL=model.pkl for Windows)
PATH_TO_TRANSFORMER=transformer.pkl (set PATH_TO_TRANSFORMER=transformer.pkl for Windows)
#####Потом запустить:
uvicorn app:app --host 127.0.0.1

#####Собрать докер:
docker build -t novokolg/online_inference:v1 .
docker run -p 80:80 novokolg/online_inference:v1

#####Запуск скрипта:
python make_request.py