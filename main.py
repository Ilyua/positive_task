import time
from fastapi import FastAPI
import pandas as pd
from my_pretty_package import get_label,clustering,vectorizer,calculate_shannon_entropy
import requests
from pydantic import BaseModel

app = FastAPI()

statistics = {'total_processed': 0}


class Vector(BaseModel):
    vector: str

@app.get("/")
def read_root():
    return 'Тестовое задание для Positive Technologies. Достигнутая точность работы - 92%'

@app.post("/get_label")
def get_vector_label(data: Vector):

    vector = data.vector.lower()
    entropy = calculate_shannon_entropy(vector)

    if 2 >= entropy >= 4:
        label = -2
    else:

        output = vectorizer.transform([vector]).toarray().flatten()
        subm_df = pd.DataFrame(data=[vector], columns=['vector'])
        subm_df.to_csv('submission_data.csv', mode='a')
        label =  get_label(clustering, output)

    statistics['total_processed'] += 1
    if statistics.get(label, None) is None:
        statistics['processed_label'+'_'+str(label)] = 0
    else:
        statistics['processed_label'+'_'+str(label)] +=1
    return f'Example label is {label}. {statistics}'
