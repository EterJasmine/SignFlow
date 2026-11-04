import {QuickSign} from './QuickSign';
import React,{useEffect,useState,useRef} from 'react';
type Entry={id:string;video_id:string;meaning:string;start_ms:number;end_ms:number;source_note:string;recognition_ready?:boolean;draft_reason?:string;video_available?:boolean};
type Source={video_id:string;filename:string;count:number;meanings:string[];duration_ms?:number};
type Candidate={start_ms:number;end_ms:number;label:string;taken:boolean;usable:boolean;reason:string|null};
type Output={matched:boolean;meaning:string|null;candidates:{meaning:string;distance:number}[];notice:string};
const PROJECT='手语示范与练习';
async function api(path:string,body?:unknown,method?:string){
 const init:RequestInit={method:method||(body?'POST':'GET')};
 if(body instanceof FormData){init.body=body}
 else if(body){init.headers={'Content-Type':'application/json'};init.body=JSON.stringify(body)}
 const r=await fetch('/api/v1'+path,init);
 const d=await r.json().catch(()=>({}));
 if(!r.ok)throw Error(typeof d.detail==='string'?d.detail:JSON.stringify(d.detail||d));
 return d;
}
const sec=(ms:number)=>(ms/1000).toFixed(2);
// 实时识别每秒会更新十几次读数，若不隔离，这个列表（含缩略图与候选区间）会跟着重渲染，
// 把主线程占满后视频与骨骼都显顿。它的入参只有练习录像，与实时状态无关。
export const SignLibrary=React.memo(function SignLibrary({clip}:{clip:string}){
 const [training,setTraining]=useState<{state:string;result?:{samples:number;classes:number;validation:number;validation_accuracy:number|null};message?:string}>({state:'idle'}),[trainingRequest,setTrainingRequest]=useState(false);
 useEffect(()=>{let stopped=false;const update=()=>api('/sign/live/training/status').then(d=>{if(!stopped)setTraining(d)}).catch(()=>{});update();const timer=setInterval(update,2000);return()=>{stopped=true;clearInterval(timer)}},[]);
 async function retrain(){setTrainingRequest(true);try{await api('/sign/live/training/retrain',{});setTraining({state:'queued'});setMessage('训练已提交，使用当前示范与已确认纠错样本。')}catch(e){setMessage((e as Error).message)}finally{setTrainingRequest(false)}}
 const player=useRef<HTMLVideoElement>(null),meaningInput=useRef<HTMLInputElement>(null);
 const [speed,setSpeed]=useState('1'),[mode,setMode]=useState<'teach'|'practice'>('teach'),[query,setQuery]=useState(''),[progress,setProgress]=useState(0);
 const [entries,setEntries]=useState<Entry[]>([]),[sources,setSources]=useState<Source[]>([]),[candidates,setCandidates]=useState<Candidate[]>([]);
 const [meaning,setMeaning]=useState(''),[note,setNote]=useState(''),[video,setVideo]=useState(''),[duration,setDuration]=useState(0);
 const [start,setStart]=useState('0'),[end,setEnd]=useState('0'),[busy,setBusy]=useState(false),[tick,setTick]=useState(0),[pending,setPending]=useState<string|null>(null);
 const [message,setMessage]=useState('先导入一个示范视频，或在下方选一条已标注的视频继续加词。'),[output,setOutput]=useState<Output|null>(null);
 const [preview,setPreview]=useState<Entry|null>(null),[previewSpeed,setPreviewSpeed]=useState('1'),[previewing,setPreviewing]=useState(false);
 const previewPlayer=useRef<HTMLVideoElement>(null),previewSource=useRef<Entry|null>(null);
 previewSource.current=preview;
 useEffect(()=>{if(player.current)player.current.playbackRate=Number(speed)},[speed,video]);
 useEffect(()=>{if(previewPlayer.current)previewPlayer.current.playbackRate=Number(previewSpeed)},[previewSpeed,preview]);
 // 示范区间常常不到一秒，timeupdate 只有 4Hz，跟不上循环边界，这里逐帧校正。
 useEffect(()=>{
  if(!preview)return;
  let raf=0;
  const step=()=>{
   const v=previewPlayer.current,p=previewSource.current;
   if(v&&p){const a=p.start_ms/1000,b=p.end_ms/1000;if(!v.paused&&(v.currentTime>=b||v.currentTime<a-.05))v.currentTime=a}
   raf=requestAnimationFrame(step);
  };
  raf=requestAnimationFrame(step);
  const onKey=(e:KeyboardEvent)=>{if(e.key==='Escape')setPreview(null)};
  window.addEventListener('keydown',onKey);
  return ()=>{cancelAnimationFrame(raf);window.removeEventListener('keydown',onKey)};
 },[preview]);
 const notified=useRef('');
 useEffect(()=>{if(!video){setCandidates([]);return}api('/sign/candidates/'+video).then(d=>{setCandidates(d.actions||[]);if(notified.current!==video){notified.current=video;if(d.actions?.length&&!d.usable_count)setMessage('这条视频的 '+d.actions.length+' 个分段都提不出有效手部特征，无法直接建示范。常见原因是手占画面太小（录屏小窗、远景）导致关键点塌缩，请让手占画面更大后重拍。')}}).catch(()=>setCandidates([]))},[video,tick]);
 async function refresh(){const [t,s]=await Promise.all([api('/sign/templates'),api('/sign/sources')]);setEntries(t);setSources(s)}
 useEffect(()=>{refresh().catch(e=>setMessage(e.message))},[]);
 function nudge(which:'start'|'end',delta:number){
  const a=Math.round(Number(start)*1000),b=Math.round(Number(end)*1000);
  if(!Number.isFinite(a)||!Number.isFinite(b))return;
  const ms=which==='start'?Math.max(0,Math.min(b-1,a+delta)):Math.min(duration,Math.max(a+1,b+delta));
  if(ms<0||ms>duration)return;
  (which==='start'?setStart:setEnd)(sec(ms));setPreviewing(false);
  if(player.current){player.current.pause();player.current.currentTime=ms/1000}
 }
 async function pickRange(from:number,to:number){setStart(sec(from));setEnd(sec(to));setOutput(null);setPreviewing(false);if(player.current){player.current.pause();player.current.currentTime=from/1000}setMessage('已取用区间 '+sec(from)+'–'+sec(to)+' 秒。填写含义后保存，起点会自动接到本段末尾。');setTimeout(()=>meaningInput.current?.focus(),0)}
 function adopt(t:Entry){
  setVideo(t.video_id);setPreview(null);setOutput(null);setPreviewing(false);setTick(x=>x+1);
  const s=sources.find(x=>x.video_id===t.video_id);if(s?.duration_ms)setDuration(s.duration_ms);
  setStart(sec(t.start_ms));setEnd(sec(t.end_ms));
  setMessage('已切换到「'+t.meaning+'」所在视频，并定位到 '+sec(t.start_ms)+'–'+sec(t.end_ms)+' 秒，可以直接改区间重存。');
 }
 async function upload(file:File){
  setBusy(true);setVideo('');setOutput(null);setProgress(0);setMeaning('');setNote('');setCandidates([]);setTick(0);
  try{
   if(file.size>80*1024*1024)throw Error('视频最大 80MB');
   setMessage('上传与骨骼分析中，请稍候…');
   let p:any=null;
   try{const list=await api('/projects');p=Array.isArray(list)?list.find((x:any)=>x.name===PROJECT):null}catch{/* 列表不可用就退化为新建 */}
   if(!p)p=await api('/projects',{name:PROJECT});
   const fd=new FormData();fd.append('file',file);
   const v=await api('/projects/'+p.id+'/videos',fd);
   setDuration(v.duration_ms);setStart('0');setEnd(sec(v.duration_ms));
   let job=await api('/videos/'+v.id+'/analysis-jobs',{});
   const deadline=Date.now()+600000;
   while(['queued','processing'].includes(job.status)){
    if(Date.now()>deadline)throw Error('分析仍在后台运行，请稍后在视频工作台查看');
    await new Promise(r=>setTimeout(r,1200));job=await api('/jobs/'+job.id);setProgress(job.progress||0)
   }
   if(job.status!=='completed')throw Error(job.error||'分析未完成');
   setVideo(v.id);setTick(1);
   setMessage('视频已就绪。若下方列出了分段建议，可直接逐个取用区间并填词；一条视频可以标注任意多个词。');
  }catch(e){setMessage((e as Error).message)}finally{setBusy(false)}
 }
 async function act(save:boolean){
  setBusy(true);setOutput(null);
  try{
   const body={video_id:video,start_ms:Math.round(Number(start)*1000),end_ms:Math.round(Number(end)*1000)};
   if(!Number.isFinite(body.start_ms)||!Number.isFinite(body.end_ms)||body.start_ms<0||body.end_ms<=body.start_ms||body.end_ms>duration)throw Error('请选择有效区间：起点需早于终点，且不能超出视频');
   if(!save){setOutput(await api('/sign/match',body));setMessage('检索完成。');return}
   if(!meaning.trim())throw Error('请填写这个动作的含义');
   const saved=await api('/sign/templates',{...body,meaning,source_note:note});
   setMeaning('');setNote('');
   // 起点接到本段末尾时终点必须一起后移，否则起止相等，播放到该点会被暂停。
   setStart(sec(body.end_ms));
   if(body.end_ms<duration)setEnd(sec(duration));
   setPreviewing(false);setTick(t=>t+1);
   await refresh();
   setMessage(saved.recognition_ready
     ?(body.end_ms<duration
       ?'已收录「'+saved.meaning+'」。起点已移到 '+sec(body.end_ms)+' 秒，终点也放到了视频末尾，可继续标注下一个词。'
       :'已收录「'+saved.meaning+'」。这段已经到视频末尾，如需再标一个词请先调整起止。')
     :'已保存为待完善示范：'+saved.draft_reason+'。补足有效手部特征后才会进入检索。');
   setTimeout(()=>meaningInput.current?.focus(),0);
  }catch(e){setMessage((e as Error).message)}finally{setBusy(false)}
 }
 async function remove(id:string,label:string){
  setBusy(true);
  try{
   const r=await api('/sign/templates/'+id,undefined,'DELETE');
   setPending(null);setTick(t=>t+1);await refresh();
   setMessage('已删除「'+label+'」。'+(r.orphan_video?'这条视频已不再承载任何示范，视频文件仍保留在工作台。':'该视频的其它示范不受影响。'));
  }catch(e){setMessage((e as Error).message)}finally{setBusy(false)}
 }
 const current=entries.filter(t=>t.video_id===video);
 const shown=entries.filter(t=>t.meaning.includes(query.trim()));
 const thumb=(t:Entry)=><img className="sign-thumb" src={'/api/v1/sign/templates/'+t.id+'/frames?count=1&height=110'} alt={'示范画面：'+t.meaning+'，'+sec(t.start_ms)+' 秒'} title="点开预览这个示范" loading="lazy" onClick={()=>setPreview(t)} onError={e=>{e.currentTarget.style.display='none'}}/>;
 return <section id="sign-library" className="panel" style={{scrollMarginTop:20}}><div className="row"><h2>我的手语示范库</h2><button className="primary" disabled={trainingRequest||['queued','running'].includes(training.state)} onClick={retrain}>{trainingRequest||training.state==='queued'?'等待训练…':training.state==='running'?'正在训练…':'重新训练'}</button></div><p role="status">{training.state==='complete'&&training.result?'最近训练：'+training.result.samples+'条样本、'+training.result.classes+'个词，验证'+training.result.validation+'条，正确率'+(training.result.validation_accuracy===null?'暂无':Math.round(training.result.validation_accuracy*100)+'%')+'。实验分类器未自动替换线上匹配。':training.state==='failed'?'最近训练失败，请检查样本数量或后台日志。':training.state==='running'?'后台训练中，当前识别可继续使用。':training.state==='queued'?'训练已排队。':''}</p><QuickSign disabled={busy} onAnnotate={async file=>{setMode('teach');await upload(file);}}/>
 <div className="tabs"><button disabled={busy} className={mode==='teach'?'selected':''} onClick={()=>{setMode('teach');setOutput(null)}}>＋ 添加示范</button><button disabled={busy} className={mode==='practice'?'selected':''} onClick={()=>{setMode('practice');setOutput(null)}}>识别我的练习</button></div>
 <p>{mode==='teach'?'一条视频可以标注多个词：取用区间 → 填含义 → 保存，起点自动接续':'选练习视频或使用上方录制 → 选择动作 → 查找相近示范'}</p>
 <div className="row">
  <label className="button">{mode==='teach'?'上传新的示范视频':'选择练习视频'}<input hidden type="file" accept="video/*" disabled={busy} onChange={e=>{const f=e.target.files?.[0];if(f)void upload(f);e.target.value=''}}/></label>
  <button disabled={!clip||busy} onClick={()=>{void fetch(clip).then(r=>r.blob()).then(b=>upload(new File([b],'camera.'+(b.type.includes('mp4')?'mp4':'webm'),{type:b.type}))).catch(e=>setMessage(e.message))}}>分析本次摄像头录制</button>
 </div>
 {sources.length>0&&<div className="row"><label>继续标注已有视频 <select aria-label="选择已用于示范的视频" value={sources.some(s=>s.video_id===video)?video:''} disabled={busy} onChange={e=>{const v=e.target.value;setVideo(v);setOutput(null);const s=sources.find(x=>x.video_id===v);if(s?.duration_ms)setDuration(s.duration_ms);setMessage(v?'已切换到「'+(s?.filename||v)+'」，下面列出了它的分段建议。':'')}}><option value="">— 选择 —</option>{sources.map(s=><option key={s.video_id} value={s.video_id}>{s.filename} · 已标 {s.count} 个词</option>)}</select></label></div>}
 <p role="status">{message}</p>{busy&&<progress aria-label="视频处理进度" max="100" value={progress}/>}
 {video&&<><video onLoadedMetadata={e=>{e.currentTarget.playbackRate=Number(speed)}} ref={player} src={'/api/v1/videos/'+video+'/media'} onTimeUpdate={e=>{const v=e.currentTarget;if(previewing&&!v.paused&&v.currentTime>=Number(end)){v.pause();setPreviewing(false)}}} controls style={{width:'100%',maxHeight:320}}/>
 <div className="row"><label>播放速度 <select aria-label="示范播放倍速" value={speed} onChange={e=>setSpeed(e.target.value)}>{[.25,.5,.75,1,1.25,1.5,2].map(v=><option key={v} value={v}>{v}×</option>)}</select></label><button disabled={busy} onClick={()=>{if(player.current)setStart(Math.min(Number(end)-.01,Math.max(0,player.current.currentTime)).toFixed(2))}}>当前位置设为起点</button><button disabled={busy} onClick={()=>{if(player.current)setEnd(Math.min(duration/1000,Math.max(Number(start)+.01,player.current.currentTime)).toFixed(2))}}>当前位置设为终点</button><button onClick={()=>{setStart('0');setEnd(sec(duration))}}>使用整段</button><button onClick={()=>{const p=player.current;if(!p)return;setPreviewing(true);p.currentTime=Number(start);p.play().catch(()=>{})}}>预览本区间</button></div>
 <div className="row"><label>起点 <input type="number" step="0.05" min="0" value={start} onChange={e=>setStart(e.target.value)} style={{width:88}}/> 秒</label><button disabled={busy} onClick={()=>nudge('start',-.1)}>−0.1</button><button disabled={busy} onClick={()=>nudge('start',.1)}>+0.1</button><label>终点 <input type="number" step="0.05" value={end} onChange={e=>setEnd(e.target.value)} style={{width:88}}/> 秒</label><button disabled={busy} onClick={()=>nudge('end',-.1)}>−0.1</button><button disabled={busy} onClick={()=>nudge('end',.1)}>+0.1</button></div>
 {mode==='teach'&&<details open={!current.length}><summary>这条视频的分段建议（{candidates.filter(c=>c.usable).length} 段可用 / 共 {candidates.length} 段）</summary>{!candidates.length?<p className="placeholder">这条视频还没有可用分段；请手动设置起止，或换一段动作更清楚的素材。</p>:<div className="row">{candidates.map((c,i)=><button key={i} disabled={busy||c.taken||!c.usable} title={c.taken?'该区间已有示范':(c.reason||'取用 '+sec(c.start_ms)+'–'+sec(c.end_ms)+' 秒')} onClick={()=>void pickRange(c.start_ms,c.end_ms)}>{c.taken?'✓ ':c.usable?'+ ':'⚠ '}{sec(c.start_ms)}–{sec(c.end_ms)}</button>)}</div>}{candidates.some(c=>!c.usable)&&<p className="hint">标 ⚠ 的分段提不出有效手部特征，鼠标悬停可看具体原因。常见是手占画面太小；也可以手动把起止收窄到动作最清楚的那一小段再试。</p>}</details>}
 {mode==='teach'&&<div className="row"><input ref={meaningInput} aria-label="这个动作的含义" placeholder="这个动作的含义，例如：谢谢" value={meaning} maxLength={100} onChange={e=>setMeaning(e.target.value)} style={{flex:1,minWidth:180}}/><input aria-label="来源备注" placeholder="来源备注（可选）" value={note} maxLength={500} onChange={e=>setNote(e.target.value)} style={{flex:1,minWidth:140}}/></div>}
 <div className="row">{mode==='teach'?<button className="primary" disabled={busy||!meaning.trim()} onClick={()=>act(true)}>保存到示范库</button>:<button className="primary" disabled={busy||!entries.length} onClick={()=>act(false)}>识别这段练习</button>}</div>
 </>}
 {output&&<div className="hint"><b>{output.matched?'候选匹配：'+output.meaning:'未可靠匹配'}</b>{output.candidates.map((c,i)=><p key={i}>{c.meaning} · 距离 {c.distance.toFixed(3)}（越小越近）</p>)}<p>{output.notice}</p></div>}
 {current.length>0&&<><h3>这条视频已标 {current.length} 个词</h3>{current.sort((a,b)=>a.start_ms-b.start_ms).map(t=><div key={t.id} className="row sign-row">{thumb(t)}<span style={{flex:1,minWidth:0}}><b>{t.meaning}</b> · {sec(t.start_ms)}–{sec(t.end_ms)} 秒{t.recognition_ready===false?'（待完善）':''}</span><span><button disabled={busy} onClick={()=>setPreview(t)}>预览</button><button disabled={busy} onClick={()=>void pickRange(t.start_ms,t.end_ms)}>定位</button>{pending===t.id?<><button disabled={busy} style={{color:'#A32D2D'}} onClick={()=>void remove(t.id,t.meaning)}>确认删除</button><button disabled={busy} onClick={()=>setPending(null)}>取消</button></>:<button disabled={busy} onClick={()=>setPending(t.id)}>删除</button>}</span></div>)}</>}
 <h3>全部示范 {entries.length} 条</h3><input aria-label="搜索示范含义" placeholder="搜索已收录的手语…" value={query} onChange={e=>setQuery(e.target.value)}/>
 {!entries.length&&<p className="placeholder">还没有示范。先添加一个动作，再录制自己的练习进行匹配。</p>}
 {shown.map(t=><div key={t.id} className="row sign-row">{thumb(t)}<span style={{flex:1,minWidth:0}}><b>{t.meaning}</b> · {sec(t.start_ms)}–{sec(t.end_ms)} 秒 · {t.source_note||'未填写来源'}{t.recognition_ready===false?'（待完善：'+(t.draft_reason||'')+'）':''}</span><span><button disabled={busy} onClick={()=>setPreview(t)}>预览</button>{pending===t.id?<><button disabled={busy} style={{color:'#A32D2D'}} onClick={()=>void remove(t.id,t.meaning)}>确认删除</button><button disabled={busy} onClick={()=>setPending(null)}>取消</button></>:<button disabled={busy} onClick={()=>setPending(t.id)}>删除</button>}</span></div>)}
 <p className="hint">删除示范不会删除视频文件：同一条视频常承载多个词。请使用不同视频测试识别结果。当前支持视频与摄像头录制，图片教程尚未接入。</p>
 {preview&&<div className="preview-mask" role="dialog" aria-modal="true" aria-label={'预览示范：'+preview.meaning} onClick={()=>setPreview(null)}>
  <div className="preview-card" onClick={e=>e.stopPropagation()}>
   <div className="row" style={{margin:'0 0 10px'}}><b style={{fontSize:16}}>{preview.meaning}</b><span className="badge">{sec(preview.start_ms)}–{sec(preview.end_ms)} 秒</span><button onClick={()=>setPreview(null)}>关闭</button></div>
   {preview.video_available===false
    ?<p className="hint">源视频已不在本机，只能看下面的三个关键帧。</p>
    :<video ref={previewPlayer} src={'/api/v1/videos/'+preview.video_id+'/media'} muted autoPlay playsInline controls
      onLoadedMetadata={e=>{e.currentTarget.currentTime=preview.start_ms/1000}}
      style={{width:'100%',maxHeight:340,background:'#221b2a',borderRadius:10}}/>}
   <img className="preview-strip" src={'/api/v1/sign/templates/'+preview.id+'/frames?count=3&height=180'}
     alt={'示范「'+preview.meaning+'」的起点、中点与终点，紫线是当时识别到的手部关键点'}
     onError={e=>{e.currentTarget.style.display='none'}}/>
   <div className="row">
    <label>播放速度 <select aria-label="预览播放倍速" value={previewSpeed} onChange={e=>setPreviewSpeed(e.target.value)}>{['0.25','0.5','0.75','1'].map(v=><option key={v} value={v}>{v}×</option>)}</select></label>
    <span className="muted">该区间循环播放</span>
    <span style={{marginLeft:'auto'}}><button disabled={busy} onClick={()=>adopt(preview)}>回到标注区修改</button></span>
   </div>
   <p className="hint" style={{marginBottom:0}}>{preview.source_note||'未填写来源'}{preview.recognition_ready===false?'。这条还进不了检索：'+(preview.draft_reason||'有效手部特征不足')+'。可以先看清动作，再决定重标或删除。':'。已参与检索。'}</p>
  </div>
 </div>}
 </section>;
});
