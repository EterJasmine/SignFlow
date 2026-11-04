"""Coordinates are explicitly 2D. Model z remains source-relative metadata."""
import math

VERSION='geometry-2.0'
def xy(p,width,height):return (p['x']*width,p['y']*height)
def distance(a,b,width=1,height=1):
    ax,ay=xy(a,width,height);bx,by=xy(b,width,height)
    return math.hypot(ax-bx,ay-by)
def angle(a,b,c,width=1,height=1):
    ax,ay=xy(a,width,height);bx,by=xy(b,width,height);cx,cy=xy(c,width,height)
    u=(ax-bx,ay-by);v=(cx-bx,cy-by);norm=math.hypot(*u)*math.hypot(*v)
    return math.degrees(math.acos(max(-1,min(1,(u[0]*v[0]+u[1]*v[1])/norm)))) if norm>1e-8 else None
def palm_scale(points,width,height):
    if len(points)<21:return None
    return max(distance(points[5],points[17],width,height),distance(points[0],points[9],width,height),1.)
def shoulder_scale(points,width,height):
    return max(distance(points[11],points[12],width,height),1.) if len(points)>12 else None
def valid(p):
    return bool(p.get('detected',True) and not p.get('low_quality',False) and all(math.isfinite(p.get(k,float('nan'))) for k in ['x','y']))
def observed_point(p):
    visibility=getattr(p,'visibility',None);presence=getattr(p,'presence',None)
    return {'x':float(p.x),'y':float(p.y),'z':float(p.z),'raw_coordinates':{'x':float(p.x),'y':float(p.y),'z':float(p.z)},
            'visibility':float(visibility) if visibility is not None else None,
            'confidence':float(visibility) if visibility is not None else None,
            'presence':float(presence) if presence is not None else None,
            'detected':True,'interpolated':False,'low_quality':False,
            'coordinate_system':'normalized_image','z_space':'model_relative_not_fused'}
