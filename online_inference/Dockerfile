FROM python:3.6
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY model.pkl /model.pkl
COPY transformer.pkl /transformer.pkl
COPY app.py /app.py

WORKDIR .

ENV PATH_TO_MODEL="/model.pkl"
ENV PATH_TO_TRANSFORMER="/transformer.pkl"

CMD ["uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000"]
