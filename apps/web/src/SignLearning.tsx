import {CoachPanel} from './CoachPanel';
import {DiagnosticPreview} from './DiagnosticPreview';
import {SignLibrary} from './SignLibrary';
import React,{useEffect,useRef,useState} from 'react';
type Heard={meaning:string;distance:number;at:number};
type Hand={side:string;suspect:boolean;points:number[][]};
type Candidate={meaning:string;distance:number;support_count?:number};
type LiveState={phase:string;hands:number;speed:number;ready:boolean;heard:Heard[];skeleton?:Hand[];word?:{diagnostic_id?:string;matched:boolean;meaning:string|null;distance:number|null;reason?:string;candidates?:Candidate[]}|null};
// MediaPipe 手部 21 点的固定连接顺序：五指链路 + 掌根横向连接。
const HAND_EDGES:[number,number][]=[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
// 推帧节拍上限：70ms 一帧约 14fps。再快只是在没有新画面时重复送同一帧，
// 而分词要靠帧间真实运动，定得太密反而让速度估计在重复帧上失真。
const SPEED_TARGET_MS=70;
export function SignLearning(){
 const preview=useRef<HTMLVideoElement>(null),board=useRef<HTMLCanvasElement>(null),stream=useRef<MediaStream|null>(null),recorder=useRef<MediaRecorder|null>(null),timer=useRef<ReturnType<typeof setTimeout>|null>(null),liveTimer=useRef<ReturnType<typeof setTimeout>|null>(null),alive=useRef(true);
 // 骨骼不放在 state 里：它每秒变十来次，走 React 会让整页（含示范库列表）跟着重渲染，
 // 主线程被占住后视频和骨骼都显顿。这里只把最新点位放进 ref，由 requestAnimationFrame 平滑绘制。
 const target=useRef<Hand[]|undefined>(undefined),shown=useRef<number[][][]|[]>([]),frame=useRef(0),running=useRef(false);
 const [enabled,setEnabled]=useState(false),[recording,setRecording]=useState(false),[clip,setClip]=useState(''),[message,setMessage]=useState('开启摄像头，录制一个完整手语动作。'),[status,setStatus]=useState('正在检查系统…');
 const [live,setLive]=useState(false),[liveState,setLiveState]=useState<LiveState|null>(null),[liveError,setLiveError]=useState('');
 const [correction,setCorrection]=useState(''),[correctionMessage,setCorrectionMessage]=useState('');
 const [manual,setManual]=useState(false),[manualBusy,setManualBusy]=useState(false);
 const [previewDiagnostic,setPreviewDiagnostic]=useState<string|null>(null);
 const [trainingState,setTrainingState]=useState('');
 useEffect(()=>{let stopped=false;const update=()=>fetch('/api/v1/sign/live/training/status').then(r=>r.json()).then(d=>{if(!stopped)setTrainingState(d.state==='running'?'后台模型训练中，当前识别可继续使用':d.state==='complete'?'后台实验模型训练完成；识别库已热更新':d.state==='failed'?'后台训练未完成；纠错示范仍已用于实时匹配':'')}).catch(()=>{});update();const timer=setInterval(update,5000);return()=>{stopped=true;clearInterval(timer)}},[]);
 const [rate,setRate]=useState(0);
 const [currentResult,setCurrentResult]=useState<NonNullable<LiveState['word']>|null>(null);
 const correctionTarget=useRef<string|undefined>(undefined);
 const diagnostic=useRef<string|undefined>(undefined);
 const resultExpiry=useRef<ReturnType<typeof setTimeout>|null>(null);
 useEffect(()=>()=>{if(resultExpiry.current)clearTimeout(resultExpiry.current)},[]);
 useEffect(()=>{alive.current=true;fetch('/api/v1/sign/status').then(r=>{if(!r.ok)throw Error();return r.json()}).then(s=>{if(alive.current)setStatus(s.message)}).catch(()=>{if(alive.current)setStatus('无法连接手语服务，请重启本地系统。')});return()=>{alive.current=false;running.current=false;if(timer.current)clearTimeout(timer.current);if(liveTimer.current){clearTimeout(liveTimer.current);liveTimer.current=null}if(frame.current)cancelAnimationFrame(frame.current);fetch('/api/v1/sign/live',{method:'DELETE'}).catch(()=>{});if(recorder.current?.state==='recording')recorder.current.stop();stream.current?.getTracks().forEach(t=>t.stop())}},[]);
 useEffect(()=>()=>{if(clip)URL.revokeObjectURL(clip)},[clip]);
 async function open(){try{const s=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720}},audio:false});if(!alive.current){s.getTracks().forEach(t=>t.stop());return}stream.current=s;setEnabled(true);if(preview.current){preview.current.srcObject=s;await preview.current.play()}setMessage('请让双手、上半身和面部完整入镜，保持光线充足。')}catch(e){stream.current?.getTracks().forEach(t=>t.stop());stream.current=null;setEnabled(false);setMessage('无法开启摄像头：'+(e as Error).message)}}
 function stop(){haltLive();fetch('/api/v1/sign/live/reset',{method:'POST'}).catch(()=>{});if(timer.current)clearTimeout(timer.current);if(recorder.current?.state==='recording')recorder.current.stop();stream.current?.getTracks().forEach(t=>t.stop());stream.current=null;if(preview.current)preview.current.srcObject=null;setEnabled(false)}
 function record(){if(!stream.current)return;try{const type=['video/webm;codecs=vp8','video/webm','video/mp4'].find(t=>MediaRecorder.isTypeSupported(t));const r=new MediaRecorder(stream.current,type?{mimeType:type}:undefined);const chunks:BlobPart[]=[];r.ondataavailable=e=>{if(e.data.size)chunks.push(e.data)};r.onstop=()=>{if(timer.current)clearTimeout(timer.current);if(!alive.current)return;setClip(URL.createObjectURL(new Blob(chunks,{type:r.mimeType})));setRecording(false);setMessage('录制完成，可以回放或下载。视频暂存在本页，未自动上传。')};r.onerror=()=>{if(alive.current){setRecording(false);setMessage('录制失败，请重新开启摄像头。')}};recorder.current=r;r.start();setRecording(true);timer.current=setTimeout(()=>{if(r.state==='recording')r.stop()},15000)}catch(e){setMessage('录制失败：'+(e as Error).message)}}
 async function stopLive(){
  haltLive();
  await fetch('/api/v1/sign/live/reset',{method:'POST'}).catch(()=>{});
 }
 function haltLive(){
  running.current=false;
  if(liveTimer.current){clearTimeout(liveTimer.current);liveTimer.current=null}
  if(frame.current){cancelAnimationFrame(frame.current);frame.current=0}
  target.current=[];shown.current=[];
  setLive(false);setRate(0);
  wipe();
 }
 function drawLoop(){
  frame.current=requestAnimationFrame(drawLoop);
  const canvas=board.current;
  if(!canvas)return;
  const box=canvas.getBoundingClientRect();
  if(!box.width||!box.height)return;
  const ratio=Math.min(2,window.devicePixelRatio||1);
  const width=Math.round(box.width*ratio),height=Math.round(box.height*ratio);
  if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height}
  const context=canvas.getContext('2d');
  if(!context)return;
  context.setTransform(ratio,0,0,ratio,0,0);
  context.clearRect(0,0,box.width,box.height);
  const hands=target.current||[];
  // 检测结果的间隔不匀（受推理耗时影响），直接重画会让骨骼一跳一跳。按 EMA 追最新点位，
  // 目标位置仍是最新一帧，只是过渡更连贯；手数或点数变了就直接换，避免插到错误的手上。
  const raw=hands.map(h=>h.points);
  if(shown.current.length!==raw.length||raw.some((pts,i)=>pts.length!==(shown.current[i]||[]).length)){
   shown.current=raw.map(pts=>pts.map(([x,y])=>[x,y]));
  }else{
   shown.current=shown.current.map((pts,i)=>pts.map(([x,y],k)=>[x+(raw[i][k][0]-x)*.55,y+(raw[i][k][1]-y)*.55]));
  }
  if(!shown.current.length)return;
  const unit=Math.max(1.6,box.width/200);
  context.lineCap='round';context.lineJoin='round';
  shown.current.forEach((points,index)=>{
   if(points.length<21)return;
   const pixels=points.map(([x,y])=>[x*box.width,y*box.height]);
   const color='#3fb8a6';
   context.beginPath();
   for(const [a,b] of HAND_EDGES){const p=pixels[a],q=pixels[b];if(!p||!q)continue;context.moveTo(p[0],p[1]);context.lineTo(q[0],q[1])}
   // 先描一层暗底：真人画面常是浅色墙壁，单靠青绿线在亮背景上会糊掉。
   context.strokeStyle='rgba(16,34,32,.5)';context.lineWidth=unit*2.8;context.stroke();
   context.strokeStyle=color;context.lineWidth=unit;context.stroke();
   context.fillStyle=color;
   for(const [x,y] of pixels){context.beginPath();context.arc(x,y,unit*1.7,0,Math.PI*2);context.fill()}
  });
 }
 function wipe(){
  const canvas=board.current;
  if(!canvas)return;
  const context=canvas.getContext('2d');
  if(!context)return;
  context.setTransform(1,0,0,1,0,0);
  context.clearRect(0,0,canvas.width,canvas.height);
 }
 async function manualAction(action:'start'|'end'|'reset'){
  setManualBusy(true);try{
   const r=await fetch('/api/v1/sign/live/'+(action==='reset'?'reset':'manual/'+action),{method:'POST'});const d=await r.json();if(!r.ok)throw Error(d.detail);
   setManual(action==='start');
   if(action==='start')setCurrentResult(null);
   if(action==='end'){diagnostic.current=d.diagnostic_id;setCurrentResult(d);if(resultExpiry.current)clearTimeout(resultExpiry.current);resultExpiry.current=setTimeout(()=>setCurrentResult(null),10000)}
   setLiveError('');
  }catch(e){setManual(false);setLiveError((e as Error).message)}finally{setManualBusy(false)}
 }
 async function coachCapture(action:'start'|'end'):Promise<string|undefined>{
  if(action==='start'){
   if(!stream.current)await open();
   if(!stream.current)throw Error('请允许摄像头访问后重试');
   if(!running.current)await startLive();
  }
  for(let attempt=0;attempt<15;attempt++){
   const r=await fetch('/api/v1/sign/live/manual/'+action,{method:'POST'});const d=await r.json();
   if(!r.ok){if(action==='start'&&r.status===409&&attempt<14){await new Promise(resolve=>setTimeout(resolve,200));continue}throw Error(d.detail||'动作采集失败')}
   setManual(action==='start');
   if(action==='end'){diagnostic.current=d.diagnostic_id;setCurrentResult(d)}
   return d.diagnostic_id;
  }
 }
 async function startLive(){setManual(false);
  if(!stream.current){setMessage('请先开启摄像头，再开始实时识别');return}
  setLiveError('');setLive(true);setRate(0);setCurrentResult(null);if(resultExpiry.current)clearTimeout(resultExpiry.current);
  target.current=[];shown.current=[];
  await fetch('/api/v1/sign/live/reset',{method:'POST'}).catch(()=>{});
  const canvas=document.createElement('canvas');
  running.current=true;
  let mark=performance.now(),counted=0;
  const tick=async()=>{
   if(!running.current)return;
   const started=performance.now();
   const video=preview.current;
   if(video&&video.videoWidth&&video.readyState>=2){
    try{
     // 512 宽足够：手部模型内部会归一化到固定输入尺寸，再大只是多花编码与传输。
     const width=Math.min(512,video.videoWidth),height=Math.round(width*video.videoHeight/video.videoWidth);
     if(canvas.width!==width){canvas.width=width;canvas.height=height}
     const context=canvas.getContext('2d');
     if(!context)throw Error('无法读取画面');
     context.drawImage(video,0,0,width,height);
     const blob=await new Promise<Blob|null>(resolve=>canvas.toBlob(resolve,'image/jpeg',.72));
     if(!blob)throw Error('无法编码画面');
     const response=await fetch('/api/v1/sign/live/frame?ts='+Math.round(performance.now()),{method:'POST',body:blob,headers:{'Content-Type':'image/jpeg'}});
     const data=await response.json().catch(()=>({}));
     if(!response.ok)throw Error(typeof data.detail==='string'?data.detail:'实时识别服务没有响应');
     if(!running.current)return;
     target.current=data.skeleton||[];
     setLiveState(data);
     if(data.phase==='recording'){setCurrentResult(null);if(resultExpiry.current)clearTimeout(resultExpiry.current)}
     if(data.word){diagnostic.current=data.word.diagnostic_id;setCorrectionMessage('');setCurrentResult(data.word);if(resultExpiry.current)clearTimeout(resultExpiry.current);resultExpiry.current=setTimeout(()=>setCurrentResult(null),3000)}
     counted++;
     const now=performance.now();
     if(now-mark>=1000){setRate(Math.round(counted*1000/(now-mark)));mark=now;counted=0}
    }catch(e){if(running.current)setLiveError((e as Error).message)}
   }
   if(!running.current)return;
   // 不用固定间隔：一帧本身的耗时就超过节拍时，固定间隔只会让请求排队、时间戳越拖越后，
   // 而时间戳是分段与速度计算的基准。这里按实际耗时补齐节拍，慢的时候就吃满吞吐。
   const spent=performance.now()-started;
   liveTimer.current=setTimeout(tick,spent>=SPEED_TARGET_MS?0:SPEED_TARGET_MS-spent);
  };
  if(!frame.current)frame.current=requestAnimationFrame(drawLoop);
  void tick();
 }
 return <>{previewDiagnostic&&<DiagnosticPreview id={previewDiagnostic} onClose={()=>setPreviewDiagnostic(null)}/>}<aside><a className="brand" href="/">✳ <span>SignFlow<small>中国手语 · 学习与练习</small></span></a><a className="nav current" href="#live-practice">摄像头练习</a><a className="nav" href="#sign-coach">手语学习小助手 ↓</a><a className="nav" href="#sign-library">我的手语示范库 ↓</a></aside><main><header>SignFlow · 手语知音</header><section className="intro"><div className="eyebrow">CHINESE SIGN LANGUAGE</div><h1>让每一次表达，更清楚一点</h1><p>先录制一个完整动作，再逐步接入标准示范、词汇识别与动作对比。</p></section><div className="status" role="status">{message}</div><div className="row"><a className="button" href="#sign-library">打开我的手语示范库 ↓</a><span>上传、随录随标、管理已有示范</span></div><div id="live-practice" className="live-workspace"><section className="panel camera-panel"><h2>我的手语练习</h2><div className="player mirror"><video ref={preview} muted playsInline style={{minHeight:240}}/><canvas ref={board} aria-hidden="true"/></div><p className="hint">预览已镜像，录制保留原始方向。每次最长 15 秒，不录音。关闭摄像头会停止采集。</p><div className="row"><button className="primary" disabled={enabled} onClick={open}>开启摄像头</button><button disabled={!enabled||recording} onClick={record}>开始录制</button><button disabled={!recording} onClick={()=>recorder.current?.stop()}>结束录制</button><button disabled={!enabled} onClick={stop}>关闭摄像头</button></div>
{clip&&<><h2>本次回放</h2><video src={clip} controls playsInline style={{width:'100%'}}/><a className="button" href={clip} download="sign-practice">下载练习视频</a></>}</section><section className="panel results-panel"><h2>实时识别结果</h2><div className="live-box">
 <div className="row" style={{margin:'0 0 10px'}}><b>实时识别</b>{live?<button onClick={()=>void stopLive()}>停止实时识别</button>:<button className="primary" disabled={!enabled} onClick={()=>void startLive()}>开始实时识别</button>}</div>
 {liveError&&<p className="hint" role="alert">{liveError}</p>}
 {!live&&<p className="hint">开始后连续取帧（约 14fps，右上角显示实际帧率），在后端用与示范库完全相同的模型与特征计算；比完一个词停住手，就会给出最接近的示范。画面上会同时叠出系统跟踪到的手部骨骼。</p>}
 {live&&<><div className="row"><button disabled={manual||manualBusy} onClick={()=>void manualAction('start')}>标记动作开始</button><button disabled={!manual||manualBusy} onClick={()=>void manualAction('end')}>结束并识别</button><button disabled={manualBusy} onClick={()=>void manualAction('reset')}>恢复自动切词</button></div><p className="hint">诊断测试：点开始后做一个完整词，再点结束。最长6.5秒；结束后保持手动模式，避免自动识别覆盖结果。</p>
  <div className="live-readout">
   <div className="live-word">{(liveState?.phase==='recording'||liveState?.phase==='manual')?'识别中…':currentResult?(currentResult.matched?currentResult.meaning:'未可靠匹配'):'等待动作'}</div>
   <div className="live-meta"><span>{(liveState?.phase==='recording'||liveState?.phase==='manual')?'正在记录…':'等待开始'}</span><span>手 {liveState?.hands??0}</span><span>速度 {liveState?.speed??0}</span><span>{rate} fps</span></div>
  </div>
  <div className="live-bar"><span style={{width:Math.min(100,(liveState?.speed??0)/3*100)+'%'}}/></div>
  <div className="live-legend"><span><i style={{background:'#3fb8a6'}}/>手部骨骼</span></div>
  {liveState&&!liveState.ready&&<p className="hint">示范库里还没有可检索的示范：需要先把动作标成示范且「识别就绪」，再去比。</p>}
  {currentResult?.candidates?.length?<p className="live-candidates">候选参考（不代表确认）：{currentResult.candidates.slice(0,3).map(c=>c.meaning+' '+c.distance.toFixed(2)+(c.support_count?'（'+c.support_count+'条示范）':'')).join(' · ')}</p>:null}
  {currentResult&&!currentResult.matched&&currentResult.reason&&<p className="hint">{currentResult.reason}</p>}
  <div className="row"><button disabled={!diagnostic.current} onClick={()=>setPreviewDiagnostic(diagnostic.current||null)}>预览最近一次动作</button><input aria-label="实际手语含义" placeholder="识别错了？填写你实际做的词" value={correction} onFocus={()=>{if(!correction.trim())correctionTarget.current=diagnostic.current}} onChange={e=>{if(!correction.trim())correctionTarget.current=diagnostic.current;setCorrection(e.target.value)}}/><button disabled={!diagnostic.current||!correction.trim()} onClick={async()=>{const id=correctionTarget.current||diagnostic.current;if(!id)return;try{const r=await fetch('/api/v1/sign/live/diagnostics/'+id+'/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({meaning:correction})});const d=await r.json();if(!r.ok)throw Error(d.detail);setCorrectionMessage(d.message);setCorrection('');correctionTarget.current=undefined}catch(e){setCorrectionMessage((e as Error).message)}}}>纠正并热更新</button></div><p role="status">{correctionMessage}</p><p role="status">{trainingState}</p><p className="hint">最近 50 次动作特征与结果仅存本机，不保存摄像头原始画面。纠正后立即加入识别库，并在后台重训实验模型。</p><p className="live-heard">历史结果：{liveState?.heard?.slice(0,5).map(h=>h.meaning+' '+h.distance.toFixed(2)).join(' · ')||'比完一个词后停一下手，结果会出现在上面。'}</p>
 </>}
</div><CoachPanel diagnosticId={diagnostic.current} onCapture={coachCapture}/></section></div><SignLibrary clip={clip}/><section className="panel"><h2>识别与测评准备</h2><p role="status">{status}</p><div className="hint">可在下方导入视频、标注含义并建立自己的示范库，再检索录制动作。当前不是通用词汇模型，不提供规范性评分。</div><h2>接下来接入</h2><p>当前：用户示范入库 → 完整动作检索。后续：经验证的示范库、连续摄像头识别与动作差异反馈。</p><p>动作相似度只反映与某条示范的差异，不等同于手语表达正确性。</p></section></main></>}
