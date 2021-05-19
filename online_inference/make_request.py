import numpy as np
import pandas as pd
import requests
import json

if __name__ == "__main__":
    data = pd.read_csv("data.csv")
    data['target'] = 0
    request_features = list(data.columns)
    print(request_features)
    for i in range(100):
        request_data = [
            x.item() if isinstance(x, np.generic) else x for x in data.iloc[i].tolist()
        ]
        print({"data": [request_data], "features": request_features})
        response = requests.get(
            "http://127.0.0.1:8000/predict/",
#            json={"data": [request_data], "features": request_features},
            data=json.dumps({"data": [request_data], "features": request_features})
        )
        print(str(json.dumps({"data": [request_data], "features": request_features})))
        print(response.status_code)
        print(response)
        print(response.json())