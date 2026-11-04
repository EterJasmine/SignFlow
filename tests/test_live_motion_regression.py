from packages.pipeline.live import LiveCapture

def hand(x,tip=0):
    points=[{'x':x+(i%5)*.01,'y':.4+(i//5)*.01,'z':0} for i in range(21)]
    for i in (4,8,12,16,20):points[i]['y']+=tip
    return {'landmarks':points}
def frame(t,items):
    return {'timestamp_ms':t,'tracks':[{'track_id':k,'hands':{'unknown':v}} for k,v in items]}
def engine(frames):
    e=object.__new__(LiveCapture);e.width=640;e.height=480;e.frames=frames;return e

def test_order_change_is_not_motion():
    e=engine([frame(0,[('a',hand(.2)),('b',hand(.7))]),frame(300,[('b',hand(.7)),('a',hand(.2))])])
    assert e.wrist_motion()==0

def test_finger_motion_without_wrist_is_active():
    e=engine([frame(0,[('a',hand(.2))]),frame(300,[('a',hand(.2,.04))])])
    assert e.wrist_motion()>.5

def test_new_identity_does_not_create_false_motion():
    e=engine([frame(0,[('a',hand(.2))]),frame(300,[('b',hand(.7))])])
    assert e.wrist_motion()==0
