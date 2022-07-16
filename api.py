from fastapi import FastAPI
from pydantic import BaseModel
from train import detect_fraud, test


class hyperParamsType(BaseModel):
    max_depth_min: int = 5
    max_depth_max: int = 10
    max_features_min: int = 5
    max_features_max: int = 10
    min_samples_leaf_min: int = 5
    min_samples_leaf_max: int = 10
    min_samples_split_min: int = 5
    min_samples_split_max: int = 10
    n_estimators_min: int = 5
    n_estimators_max: int = 10


app = FastAPI(title='Insurance Fraud',
              description='Insurance Claim Fraud Detection', )


@app.get("/", tags=["Train_Test"])
def read_root():
    return "Application Started"


@app.post("/train", tags=["Train_Test"])
async def create_score(params: hyperParamsType):
    result = detect_fraud(params)
    return result


@app.get("/test", tags=["Train_Test"])
async def Result():
    result = test()
    return result
