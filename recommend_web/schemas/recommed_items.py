from typing import Optional
from pydantic import BaseModel,EmailStr


#properties required during user creation
class ItemCreate(BaseModel):
    keyword : str 