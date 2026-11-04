from typing import Literal
from uuid import uuid4
from pydantic import BaseModel,Field,model_validator

TAXONOMY={'static':['unknown','other','open_palm','fist','point','v_sign','thumb_up','pinch','heart'],
          'dynamic':['unknown','other','wave','spread','approach','cross','rotate','point_cheek','raise'],
          'phase':['unknown','preparation','transition','hold','release']}
class Annotation(BaseModel):
    id:str=Field(default_factory=lambda:'annotation_'+uuid4().hex)
    kind:Literal['static','dynamic','phase']
    label:str
    start_ms:int=Field(ge=0)
    end_ms:int=Field(gt=0)
    person_id:str=Field(default='',max_length=100)
    hand:Literal['left','right','both','unknown']='unknown'
    body_reference:Literal['face','shoulders','chest','none','unknown']='unknown'
    action_id:str=Field(default='',max_length=100)
    boundary_source:Literal['action_seed','manual']='action_seed'
    reviewed:bool=False
    annotator:str=Field(default='local',max_length=100)
    @model_validator(mode='after')
    def validate_annotation(self):
        if self.end_ms<=self.start_ms:raise ValueError('标注区间必须为正')
        if self.label not in TAXONOMY[self.kind]:raise ValueError('标签不在当前 taxonomy 内')
        return self
