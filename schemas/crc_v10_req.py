from pydantic import BaseModel

class CRCV10Req(BaseModel):
    data: list[dict]
