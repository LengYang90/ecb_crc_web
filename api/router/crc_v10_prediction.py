import json
import os
from fastapi import APIRouter, FastAPI
from schemas.crc_v10_req import CRCV10Req
from scripts.MLGenie_prediction import MLGeniePredictor
from fastapi.responses import JSONResponse

router = APIRouter()


@router.post("/predict")
def crc_v1_0_predict(req: CRCV10Req):
    try:
        print("req.data: ", req.data)
        #print(req.data)
        predictor = MLGeniePredictor(req.data)
        pred_result = predictor.run()
        return pred_result
    except Exception as e:
        print("error: ", e)
        return str(e)



    



