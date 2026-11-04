from packages.pipeline.quality import usable_for_recognition

def hand():
 return {'landmarks':[{'x':.2+i*.01,'y':.4} for i in range(21)],'quality':{'suspect':True,'reasons':['scale_jump'],'geometry':{'palm_px':30,'finger_ratio_max':1.2,'knuckle_spread':.2}}}
def test_scale_warning_still_usable():assert usable_for_recognition(hand())
def test_degenerate_remains_rejected():
 h=hand();h['quality']['reasons']=['degenerate'];assert not usable_for_recognition(h)
def test_nonfinite_is_rejected():
 h=hand();h['landmarks'][0]['x']=float('nan');assert not usable_for_recognition(h)
