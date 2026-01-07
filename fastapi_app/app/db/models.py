from sqlalchemy import Column, Integer, DateTime, LargeBinary, Boolean
from sqlalchemy.sql import func
from sqlalchemy import Index
from .database import Base

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    image_png = Column(LargeBinary, nullable=False)

    predicted_label = Column(Integer, nullable=False)
    true_label = Column(Integer, nullable=True)

    corrected = Column(Boolean, default=False, nullable=False)

Index("ix_predictions_created_at", Prediction.created_at)
