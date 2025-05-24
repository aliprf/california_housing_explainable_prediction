
from pydantic import BaseModel


class CaliforniaHousingModel(BaseModel):
    MedInc: float       
    HouseAge: float     
    AveRooms: float     
    AveBedrms: float    
    Population: float   
    AveOccup: float     
    Latitude: float     
    Longitude: float 