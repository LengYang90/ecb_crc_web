from fastapi import APIRouter

from api.router.crc_v10_prediction import router as crc_v10_router

router = APIRouter()

router.include_router(crc_v10_router, prefix="/crc", tags=["predict"])
