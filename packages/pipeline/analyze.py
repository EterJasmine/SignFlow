from pathlib import Path
from contextlib import ExitStack
import json, math,os,hashlib
from collections import Counter
import cv2
import numpy as np
from .scene import evidence, summarize_scene,pipeline_for
from .motion import motion_steps
from .tracking import HandAssociator
from .teaching import enrich_action,trajectory_svg
from .geometry import observed_point
from .occlusion import OcclusionRecovery,map_crop,merge_recovered,search_regions
from .tracking_v2 import PredictiveHands
from .quality import assess, pose_assess, hand_reference
from .trajectory import process_trajectories
from .features import extract_features
from .segmentation import segment_actions
from .composer import compose_step,diagram_viewbox

ROOT=Path(__file__).resolve().parents[2]

class Cancelled(Exception): pass

def points(landmarks):
    return [observed_point(p) for p in landmarks]

CLAHE=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
def prepare_frame(bgr,max_side):
    """缩放到检测分辨率，并在亮度通道上做局部对比度均衡。
    暗调、逆光、纯色背景的素材上这一步能把「全可信手部帧」从 59% 提到 66%，并消掉幽灵手。"""
    scale=min(1,max_side/max(bgr.shape[:2]))
    img=cv2.resize(bgr,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
    sharpness=float(cv2.Laplacian(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var())
    if os.getenv('GESTUREFLOW_ENHANCE','1')!='0':
        lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB);l,a,b=cv2.split(lab)
        img=cv2.cvtColor(cv2.merge([CLAHE.apply(l),a,b]),cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB),sharpness

def credible_poses(poses,width,height,hands,stats):
    """丢掉不像人的骨架。

    近景手部、纯色暗背景上 MediaPipe 会把手指映射成四肢，给出 visibility≈1 但整体塌缩的
    「人」。这类结果如果留着，会污染取景分类、手-人归属与尺度归一化，所以按几何比例逐帧丢弃。
    hands 是同一帧的手部关键点，用作参照尺度（没有手时可省略）。
    """
    palm_ref,box_ref=hand_reference([{'landmarks':h} for h in hands],width,height)
    kept=[]
    for pose in poses:
        verdict=pose_assess(pose,width,height,palm_ref,box_ref)
        stats['observed']+=1
        if verdict['plausible']:kept.append(pose);continue
        stats['rejected']+=1
        for reason in verdict['reasons']:stats['reasons'][reason]=stats['reasons'].get(reason,0)+1
    return kept

class Tracker:
    """Nearest-center baseline, reset at each shot; not ByteTrack."""
    def __init__(self):self.previous={};self.counter=0
    def update(self, poses):
        remaining=set(self.previous);result=[];next_state={}
        for pose in poses:
            center=np.mean([[pose[i]['x'],pose[i]['y']] for i in [11,12,23,24]],axis=0)
            match=min(remaining,key=lambda k:np.linalg.norm(center-self.previous[k]),default=None)
            if match is not None and np.linalg.norm(center-self.previous[match])<.25:
                ident=match;remaining.remove(match)
            else:self.counter+=1;ident=f'person_{self.counter:04}'
            next_state[ident]=center;result.append({'track_id':ident,'pose_landmarks':pose,'hands':{},'face_landmarks':[]})
        self.previous=next_state
        return result

def diagram(frame, width=640, height=640):
    hand_edges=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]
    body_edges=[(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28)]
    lines=[]
    for track in frame['tracks']:
        for p,edges,color in [(track['pose_landmarks'],body_edges,'#9b82c3')]+[(h['landmarks'],hand_edges,'#39a89b') for h in track['hands'].values()]:
            for a,b in edges:
                if max(a,b)>=len(p):continue
                if any(p[k].get('low_quality') or (p[k].get('visibility') is not None and p[k]['visibility']<.5) for k in [a,b]):continue
                lines.append(f'<line x1="{p[a]["x"]*width:.2f}" y1="{p[a]["y"]*height:.2f}" x2="{p[b]["x"]*width:.2f}" y2="{p[b]["y"]*height:.2f}" stroke="{color}" stroke-width="3"/>')
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="#f7f4fb"/>'+''.join(lines)+'</svg>'

def analyze_video(video_path, output, video_id, report, cancel=lambda:False,checkpoint_directory=None,scene_overrides=None):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from scenedetect import detect, ContentDetector
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    models=ROOT/'models'
    required=['hand_landmarker.task','pose_landmarker.task','face_landmarker.task']
    missing=[m for m in required if not (models/m).exists()]
    if missing:raise RuntimeError('模型缺失，请运行 python scripts/download_models.py：'+', '.join(missing))
    report('scene_detection',5)
    cap=cv2.VideoCapture(str(video_path))
    fps=cap.get(cv2.CAP_PROP_FPS);count=cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if fps<=0 or count<=0:cap.release();raise ValueError('无法读取视频元数据')
    duration=round(count/fps*1000);w=int(cap.get(3));h=int(cap.get(4))
    if duration>120000 or w*h>3840*2160:cap.release();raise ValueError('首版支持 120 秒以内、最高 4K 视频')
    try:
        pose_stats={'observed':0,'rejected':0,'reasons':{}}
        probe_pose_stats={'observed':0,'rejected':0,'reasons':{}}
        quality_stats={'observed':0,'suspect':0,'suspect_frames':0}
        checkpoint_root=Path(checkpoint_directory) if checkpoint_directory else output
        checkpoint_root.mkdir(parents=True,exist_ok=True)
        checkpoint=checkpoint_root/'vision.json'
        scene_overrides=[{key:s[key] for key in ['start_ms','end_ms','framing_type']} for s in scene_overrides or []]
        stat=Path(video_path).stat()
        code_digest=hashlib.sha256(b''.join(p.read_bytes() for p in sorted(Path(__file__).parent.glob('*.py')))).hexdigest()
        signature={'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,'code':code_digest,'models':{p.name:p.stat().st_mtime_ns for p in models.glob('*.task')},'scene_overrides':scene_overrides or []}
        cached=None
        if checkpoint.exists():
            try:
                payload=json.loads(checkpoint.read_text(encoding='utf8'))
                if payload.get('signature')==signature:cached=payload
            except (ValueError,OSError):pass
        if cached:
            frames=cached['frames'];scenes=cached['scenes'];step=cached['step'];report('resume_vision',85)
            pose_stats={**pose_stats,**(cached.get('pose_stats') or {})}
            probe_pose_stats={**probe_pose_stats,**(cached.get('probe_pose_stats') or {})}
            quality_stats={**quality_stats,**(cached.get('quality_stats') or {})}
        else:
            cuts=detect(str(video_path),ContentDetector(threshold=30,min_scene_len=max(1,int(fps*.6))))
            boundaries=sorted({0,duration,*[round(s.get_seconds()*1000) for s,_ in cuts]})
            report('scene_probe',8)
            probes=[]
            probe_pose_stats={'observed':0,'rejected':0,'reasons':{}}
            # First pass: independent 2fps probes establish stable shot framing.
            # 这里必须用视频模式（detect_for_video）：静态 detect 在每个采样点上从头找骨架，
            # 真人素材上会给出大量塌缩的假骨架，纯手素材则两种模式都是假骨架。
            with vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_buffer=(models/required[0]).read_bytes()),running_mode=vision.RunningMode.VIDEO,num_hands=4)) as probe_hands, vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_buffer=(models/required[1]).read_bytes()),running_mode=vision.RunningMode.VIDEO,num_poses=4)) as probe_pose:
                for t in range(0,duration,500):
                    if cancel():raise Cancelled()
                    cap.set(cv2.CAP_PROP_POS_MSEC,t);ok,bgr=cap.read()
                    if not ok:continue
                    rgb,_=prepare_frame(bgr,640);ph,pw=rgb.shape[:2]
                    image=mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb))
                    pr=probe_pose.detect_for_video(image,t);hr=probe_hands.detect_for_video(image,t)
                    probe_hand_points=[points(h) for h in hr.hand_landmarks]
                    poses=credible_poses([points(p) for p in pr.pose_landmarks],pw,ph,probe_hand_points,probe_pose_stats)
                    si=max(i for i,x in enumerate(boundaries[:-1]) if x<=t)
                    probes.append({'scene_id':f'scene_{si+1:04}','evidence':evidence(poses,hr.hand_landmarks)})
            scenes=[summarize_scene([f for f in probes if f['scene_id']==f'scene_{i+1:04}'],start,end,f'scene_{i+1:04}') for i,(start,end) in enumerate(zip(boundaries,boundaries[1:]))]
            for scene in scenes:
                for override in scene_overrides or []:
                    overlap=max(0,min(scene['end_ms'],override['end_ms'])-max(scene['start_ms'],override['start_ms']))
                    if overlap/max(1,scene['end_ms']-scene['start_ms'])>.8:
                        scene['automatic_framing_type']=scene['framing_type'];scene['framing_type']=override['framing_type'];scene['classification_method']='manual';scene['review_status']='approved';scene['pipeline']=pipeline_for(scene['framing_type'],scene['person_count']);break
            cap.set(cv2.CAP_PROP_POS_FRAMES,0)
            report('landmark_detection',12)
            frames=[];tracker=Tracker();hand_tracker=PredictiveHands(w,h);previous_scene=None
            palms={}
            # Second pass uses the selected shot strategy; unknown always runs pose.
            with ExitStack() as stack:
                hands=stack.enter_context(vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_buffer=(models/required[0]).read_bytes()),running_mode=vision.RunningMode.VIDEO,num_hands=4,min_hand_detection_confidence=.5)))
                pose=stack.enter_context(vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_buffer=(models/required[1]).read_bytes()),running_mode=vision.RunningMode.VIDEO,num_poses=4,min_pose_detection_confidence=.5)))
                recovery=OcclusionRecovery()
                recovery_detector=None
                region_detector=None
                last_region_search=-10000
                region_searches=0;region_recovered=0
                face=None
                step=max(1,round(fps/8));index=0
                while True:
                    ok,bgr=cap.read()
                    if not ok:break
                    if cancel():raise Cancelled('任务已取消')
                    if index%step and round(index/fps*1000)>recovery.dense_until:index+=1;continue
                    t=round(index/fps*1000);index+=1
                    scene_index=max(i for i,x in enumerate(boundaries[:-1]) if x<=t)
                    scene_id=f'scene_{scene_index+1:04}'
                    if previous_scene!=scene_id:tracker.previous={};hand_tracker.reset();recovery.reset();palms={};previous_scene=scene_id
                    scale=min(1,960/max(w,h));rgb,sharpness=prepare_frame(bgr,960)
                    image=mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb))
                    strategy=scenes[scene_index]['framing_type']
                    hr=hands.detect_for_video(image,t)
                    hand_data=[points(p) for p in hr.hand_landmarks]
                    pr=pose.detect_for_video(image,t) if strategy!='hands_only' else None
                    raw_poses=[points(p) for p in pr.pose_landmarks] if pr else []
                    poses=credible_poses(raw_poses,w,h,hand_data,pose_stats)
                    tracks=tracker.update(poses)
                    ev=evidence(poses,hand_data)
                    detected_hands=[]
                    for j,hand in enumerate(hand_data):
                        label=hr.handedness[j][0]
                        detected_hands.append({'detected':True,'landmarks':hand,'model_handedness':label.category_name,'handedness_score':float(label.score)})
                    recovery.observe(detected_hands,t,w,h)
                    box=(hand_tracker.predicted_region(t) or recovery.crop_box(t,w,h)) if t<=recovery.dense_until else None
                    if box:
                        if recovery_detector is None:
                            recovery_detector=stack.enter_context(vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_buffer=(models/required[0]).read_bytes()),running_mode=vision.RunningMode.IMAGE,num_hands=4,min_hand_detection_confidence=.5)))
                        x,y,right,bottom=box
                        crop,_=prepare_frame(bgr[y:bottom,x:right],1280)
                        recovery.attempts+=1
                        recovered=recovery_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(crop)))
                        candidates=[]
                        for j,landmarks in enumerate(recovered.hand_landmarks):
                            label=recovered.handedness[j][0]
                            candidates.append({'detected':True,'landmarks':map_crop(points(landmarks),box,w,h),'model_handedness':label.category_name,'handedness_score':float(label.score),'detection_source':'source_roi'})
                        detected_hands,added=merge_recovered(detected_hands,candidates,w,h)
                        recovery.recovered+=added
                    # Search beyond known wrists: a never-detected second hand has no predicted ROI.
                    if len(detected_hands)==1 and t-last_region_search>=250:
                        last_region_search=t
                        if region_detector is None:
                            region_detector=stack.enter_context(vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_buffer=(models/required[0]).read_bytes()),running_mode=vision.RunningMode.IMAGE,num_hands=4,min_hand_detection_confidence=.3)))
                        for box in search_regions(w,h):
                            x,y,right,bottom=box
                            # Keep the original pixels and photometry for this independent retry.
                            crop=np.ascontiguousarray(cv2.cvtColor(bgr[y:bottom,x:right],cv2.COLOR_BGR2RGB))
                            found=region_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=crop))
                            region_searches+=1;candidates=[]
                            for j,landmarks in enumerate(found.hand_landmarks):
                                label=found.handedness[j][0]
                                candidates.append({'detected':True,'landmarks':map_crop(points(landmarks),box,w,h),'model_handedness':label.category_name,'handedness_score':float(label.score),'detection_source':'regional_search'})
                            detected_hands,added=merge_recovered(detected_hands,candidates,w,h)
                            region_recovered+=added
                            if len(detected_hands)>=2:break
                    tracks=hand_tracker.assign(tracks,detected_hands,t)
                    # 逐手几何自检：置信度高的错骨架只能从几何上看出来（掌心拉长、五指挤成一团）。
                    for tr in tracks:
                        for side,hand in tr['hands'].items():
                            if hand.get('interpolated'):continue
                            palm_key=(tr['track_id'],side)
                            quality=assess(hand['landmarks'],w,h,palms.get(palm_key))
                            hand['quality']=quality
                            if quality['geometry'] and not quality['suspect']:palms[palm_key]=quality['geometry']['palm_px']
                            quality_stats['observed']+=1
                            if quality['suspect']:quality_stats['suspect']+=1
                    if any(h.get('quality',{}).get('suspect') for tr in tracks for h in tr['hands'].values()):
                        quality_stats['suspect_frames']+=1
                    if strategy=='upper_body':
                        if face is None:face=stack.enter_context(vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_buffer=(models/required[2]).read_bytes()),running_mode=vision.RunningMode.VIDEO,num_faces=4)))
                        fr=face.detect_for_video(image,t)
                        for fl in fr.face_landmarks:
                            fp=points(fl);center=fp[1];body_tracks=[tr for tr in tracks if tr['pose_landmarks']]
                            nearest=min(body_tracks,key=lambda tr:math.hypot(tr['pose_landmarks'][0]['x']-center['x'],tr['pose_landmarks'][0]['y']-center['y']),default=None)
                            if nearest:nearest['face_landmarks']=fp
                    frames.append({'timestamp_ms':t,'scene_id':scene_id,'evidence':ev,'tracks':tracks,'coordinate_system':'normalized_image','image_size':{'width':w,'height':h},'sharpness':sharpness})
                    if len(frames)%8==0:report('landmark_detection',min(85,12+int(t/duration*73)))
            temp=checkpoint.with_suffix('.tmp');temp.write_text(json.dumps({'signature':signature,'frames':frames,'scenes':scenes,'step':step,'pose_stats':pose_stats,'probe_pose_stats':probe_pose_stats,'quality_stats':quality_stats},ensure_ascii=False,allow_nan=False),encoding='utf8');temp.replace(checkpoint)
        report('motion_segmentation',88)
        processed_frames,trajectory_quality=process_trajectories(frames,w,h)
        baseline_actions=motion_steps(frames,scenes,duration)
        features=extract_features(processed_frames,w,h)
        actions,action_phases,segmentation=segment_actions(processed_frames,scenes,duration,features)
        from training.inference import predict,fuse_boundaries
        model_assistance=predict(features,processed_frames,w,h)
        mode=os.getenv('GESTUREFLOW_SEGMENTER','rules')
        if mode=='baseline':actions=baseline_actions
        elif mode=='model':
            fused=fuse_boundaries(actions,model_assistance,processed_frames)
            if fused is actions:
                # 模型没接上（文件缺失/特征版本不符）就如实标成 rules，别让页面显示"模型分段"。
                mode='rules';segmentation['model_fallback']=model_assistance.get('reason')
            else:actions=fused
        segmentation['selected_mode']=mode
        for action in actions:enrich_action(action,processed_frames)
        report('rendering',94)
        for action in actions:
            if cancel():raise Cancelled()
            t=action['keyframe_ms'];cap.set(cv2.CAP_PROP_POS_MSEC,t);ok,bgr=cap.read()
            stem=action['id']
            for label,tm in [('start',action.get('start_frame_ms',action['start_ms'])),('end',action.get('end_frame_ms',action['end_ms']-1))]:
                cap.set(cv2.CAP_PROP_POS_MSEC,tm);valid_frame,pixels=cap.read()
                if valid_frame:
                    success,encoded_frame=cv2.imencode('.jpg',pixels)
                    if success:encoded_frame.tofile(str(output/(stem+'_'+label+'.jpg')));action['assets'][label+'_source']=f'/assets/{video_id}/{stem}_{label}.jpg'
            if ok:
                image_path=output/(stem+'.jpg')
                ok,encoded=cv2.imencode('.jpg',bgr)
                if ok:encoded.tofile(str(image_path));action['assets']['source']=f'/assets/{video_id}/{stem}.jpg'
            frame=min(processed_frames,key=lambda f:abs(f['timestamp_ms']-t),default=None)
            if frame:
                svg=diagram(frame,w,h).replace('</svg>',trajectory_svg(action.get('trajectories',[]),w,h,action.get('rotation_facts',[]))+'</svg>')
                kind=next((s['framing_type'] for s in scenes if s['start_ms']<=t<s['end_ms']),'unknown')
                svg,action['diagram_transform']=diagram_viewbox(svg,frame,kind,w,h)
                (output/(stem+'.svg')).write_text(svg,encoding='utf-8');action['assets']['diagram']=f'/assets/{video_id}/{stem}.svg'
        coverage={'sampled_frames':len(frames),'hand_frames':sum(any(t['hands'] for t in f['tracks']) for f in frames),'pose_frames':sum(any(t['pose_landmarks'] for t in f['tracks']) for f in frames)}
        coverage['hand_quality']={**quality_stats,
            'suspect_rate':round(quality_stats['suspect']/quality_stats['observed'],3) if quality_stats['observed'] else 0.0,
            'method':'geometry_v1','enhance':os.getenv('GESTUREFLOW_ENHANCE','1')!='0'}
        rejected=pose_stats['rejected']+probe_pose_stats['rejected']
        coverage['pose_quality']={**pose_stats,
            'reject_rate':round(pose_stats['rejected']/pose_stats['observed'],3) if pose_stats['observed'] else 0.0,
            'probe':{'observed':probe_pose_stats['observed'],'rejected':probe_pose_stats['rejected'],
                     'reasons':probe_pose_stats['reasons']},
            'rejected_total':rejected,'method':'skeleton_ratio_v1'}
        warnings=[]
        if not coverage['hand_frames']:warnings.append('采样画面中没有检测到手部；请核对手是否过小、模糊或被遮挡。')
        if not coverage['pose_frames']:
            warnings.append('没有检测到可靠人体骨骼；纯手部视频仍可显示手部关键点。')
        elif rejected:
            warnings.append('共丢弃 {} 个不符合人体比例的骨架：手部特写、纯色暗背景上 MediaPipe 常把手指当成四肢，检出「置信度很高但比例不可能」的人。'.format(rejected))
        if coverage['hand_quality']['suspect_rate']>=.3:
            warnings.append('约 {:.0%} 的手部关键点几何上不自洽（双手交叠、袖口遮挡或逆光），已标为橙色；这些片段的分段结果建议人工核对。'.format(coverage['hand_quality']['suspect_rate']))
        result={'video_id':video_id,'schema_version':'1.0','pipeline_version':'gestureflow-0.5.0',
                'duration_ms':duration,'width':w,'height':h,'fps':fps,'sample_fps':fps/step,
                'scenes':scenes,'frames':frames,'processed_frames':processed_frames,'trajectory_quality':trajectory_quality,'actions':actions,'baseline_actions':baseline_actions,'features':features,'action_phases':action_phases,'segmentation':segmentation,'lyrics':[],'coverage':coverage,'warnings':warnings,
                'coordinate_metadata':{'raw':'normalized_image','geometry':'pixel_aspect_correct','normalization':'palm_or_shoulder','z':'model_relative_not_fused'},'model_assistance':model_assistance,'checkpoint_info':{'vision_reused':cached is not None,'code_signature':code_digest},
                'limitations':['取景分类采用 2fps 探测和镜头内可见性多数投票，同一长镜头内构图变化尚未单独切分。',
                              '多人使用中心点近邻关联，非 ByteTrack，交叉遮挡后 ID 可能变化。',
                              '纯手部使用空间近邻关联，解剖学左右手未确认，交叉和缺失时不保证连续身份。',
                              '双手交叠、袖口遮挡、逆光或极低对比度时，单帧检测可能给出置信度很高但几何不可能的骨架；这类手部标 quality.suspect（界面显示橙色），几何判据只看比值，无法识别"比值正常但指到了另一只手"的情况。',
                              '人体骨架按肩宽/画面宽、躯干、髋、上臂的比例与手部尺度比对后逐帧过滤：只有手、纯色背景的画面常被 MediaPipe 检出「人」，这类骨架已丢弃，但真人骨架在极端姿态或严重遮挡下也可能被丢弃。',
                              '动作边界为腕部速度峰值候选，需人工校正；不代表语义识别结果。']}
        result['tutorial_steps']=[compose_step(a,result) for a in actions]
        (output/'result.json').write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8')
        return result
    finally:cap.release()
