// Tags
function addTag(){
  const inp=document.getElementById('in-prompt');
  const v=inp.value.trim();
  if(!v||tags.includes(v)) return;
  tags.push(v);renderTags();inp.value='';inp.focus();
}
function rmTag(i){tags.splice(i,1);renderTags()}
function renderTags(){
  document.getElementById('tag-list').innerHTML=tags.map((t,i)=>
    `<span class="tag">${esc(t)}<span class="x" onclick="rmTag(${i})">&times;</span></span>`
  ).join('');
}
function autoPrompts(){
  tags=autoExtract();renderTags();
  if(tags.length) log('Auto-extracted: '+tags.join(', '),'inf');
}
function autoExtract(){
  const txt=document.getElementById('in-instr').value;
  if(!txt) return [];
  const stop=new Set('pick up the and place it on put in into a an to from move grab take get set drop both all each'.split(' '));
  const words=txt.toLowerCase().replace(/,/g,' ').split(/\s+/);
  const out=[];let cur=[];
  for(const w of words){if(stop.has(w)){if(cur.length){out.push(cur.join(' '));cur=[]}}else cur.push(w)}
  if(cur.length) out.push(cur.join(' '));
  return out;
}

// Steps
function setStep(id){
  const order=['cap','det','plan','exec'];
  const idx=order.indexOf(id);
  order.forEach((s,i)=>{
    const el=document.getElementById('s-'+s);
    el.className='step'+(i<idx?' ok':i===idx?' act':'');
  });
}

// Log
// ts (optional): seconds since the epoch, from the server, marking when the
// event ACTUALLY happened. Without it every line is stamped at render time,
// which is why a batch of progress entries used to render with one identical
// timestamp -- the clock being read was the browser's, at paint, not the
// robot's, at the step. Callers with a real event time should pass it.
function log(msg,cls='',ts=null){
  const el=document.getElementById('log');
  const t=new Date(ts!=null?ts*1000:Date.now()).toLocaleTimeString('en',{hour12:false});
  const e=document.createElement('div');e.className='log-e fade';
  e.innerHTML=`<span class="log-t">${t}</span><span class="log-m ${cls}">${esc(msg)}</span>`;
  el.appendChild(e);el.scrollTop=el.scrollHeight;
  while(el.children.length>300) el.removeChild(el.firstChild);
}
function clearLog(){document.getElementById('log').innerHTML='';log('Cleared','inf')}

// History
function addHist(d){
  const list=document.getElementById('hist-list');
  const em=list.querySelector('.empty-msg');if(em) em.remove();
  const el=document.createElement('div');el.className='hist fade';
  el.innerHTML=`<span class="hist-dot ${d.success?'ok':'fail'}"></span>`
    +`<div class="hist-body"><div class="hist-instr">${esc(d.instruction)}</div>`
    +`<div class="hist-meta">${(d.duration||0).toFixed(1)}s | ${(d.detections||[]).length} obj</div></div>`;
  list.prepend(el);
}

// Episode bundle review
// Renders a banner above the live log with Save / Discard buttons
// for the cache from the most recent episode. Save renames the
// cache folder to episodes_kept/<task>_<id>/; Discard wipes it.
// Starting a new task auto-discards (the server clears the cache
// at the top of execute_approved).
function showEpisodePanel(instruction,ep,success){
  let bar=document.getElementById('episode-bar');
  if(!bar){
    bar=document.createElement('div');
    bar.id='episode-bar';
    bar.style.cssText='margin:8px 0;padding:10px 12px;border-radius:8px;'
      +'background:var(--bg-2,#1a1d23);border:1px solid var(--border,#2a2f3a);'
      +'display:flex;align-items:center;gap:10px;flex-wrap:wrap;font-size:13px;';
    const host=document.getElementById('sec-plan')||document.body;
    host.parentNode.insertBefore(bar,host);
  }
  const dot=success?'ok':'fail';
  bar.innerHTML=`<span class="hist-dot ${dot}"></span>`
    +`<div style="flex:1;min-width:200px;">`
    +`  <div><b>Episode cached:</b> ${esc(instruction||ep.instruction||'')}</div>`
    +`  <div style="opacity:.7;">${ep.n_samples||0} traj samples, ${ep.episode_id||''}</div>`
    +`</div>`
    +`<input id="ep-name" placeholder="task name (optional)" style="flex:1;min-width:120px;padding:4px 8px;">`
    +`<button class="btn btn-s" onclick="saveEpisode('pass')" style="background:var(--grn-15);border-color:var(--grn);color:var(--grn)">PASS</button>`
    +`<button class="btn btn-s" onclick="markFail()" style="background:var(--red-15);border-color:var(--red);color:var(--red)">FAIL</button>`
    +`<button class="btn btn-s" onclick="discardEpisode()" title="Throw the episode away entirely - prefer FAIL, failures are training data">Discard</button>`;
  const nm=document.getElementById('ep-name');
  if(nm) nm.value=instruction||'';
}
// PASS = the operator's verdict, which is the only trustworthy one right now:
// task-success predicates are not reliable enough to label a corpus with.
// Stops the run first, then records the episode as a success.
async function doPass(){
  log('PASS - stopping and recording as SUCCESS','ok');
  await doStop({silent:true});
  await saveEpisode('pass');
}
// A FAILED run is DATA, not rubbish: it is the negative half a verifier study
// needs and the expensive half to collect on purpose. Keep it, labelled.
async function markFail(){ await saveEpisode('fail'); }
async function saveEpisode(outcome){
  const nm=(document.getElementById('ep-name')||{}).value||'';
  try{
    const r=await api('POST','/api/episode/save',{task_name:nm,outcome:outcome||null});
    const tag=outcome?(' ['+outcome.toUpperCase()+']'):'';
    if(r.episode&&r.episode.kept_path){
      toast('Saved'+tag+': '+r.episode.kept_path, outcome==='fail'?'':'ok', 4000);
    }else{
      toast('Saved'+tag, outcome==='fail'?'':'ok', 4000);
    }
    const bar=document.getElementById('episode-bar');if(bar)bar.remove();
  }catch(e){toast('Save failed: '+e.message,'err',5000)}
}
async function discardEpisode(){
  try{
    await api('POST','/api/episode/discard',{});
    toast('Episode discarded','',3000);
    const bar=document.getElementById('episode-bar');if(bar)bar.remove();
  }catch(e){toast('Discard failed: '+e.message,'err',4000)}
}

// BT library view
// Lists every saved BT with success counters and a Promote toggle.
// Promoted entries are always injected into the planner's few-shot
// context (regardless of jaccard similarity) so they act as
// canonical recipes the LLM can rely on.
async function showLibrary(){
  let panel=document.getElementById('lib-panel');
  if(!panel){
    panel=document.createElement('div');
    panel.id='lib-panel';
    panel.style.cssText='position:fixed;right:12px;bottom:12px;width:380px;max-height:60vh;'
      +'overflow:auto;background:var(--bg-1,#11131a);border:1px solid var(--border,#2a2f3a);'
      +'border-radius:8px;padding:10px;z-index:9999;font-size:12px;';
    document.body.appendChild(panel);
  }
  panel.innerHTML='<div style="display:flex;justify-content:space-between;align-items:center;">'
    +'<b>BT Library</b>'
    +'<button class="btn btn-s" onclick="document.getElementById(\'lib-panel\').remove()">close</button>'
    +'</div><div id="lib-list">loading...</div>';
  try{
    const r=await api('GET','/api/library/list');
    const list=document.getElementById('lib-list');
    if(!r.entries||!r.entries.length){list.textContent='(empty, no saved BTs yet)';return}
    list.innerHTML='';
    r.entries.forEach(e=>{
      const row=document.createElement('div');
      row.style.cssText='padding:6px 0;border-top:1px solid var(--border,#222);';
      const star=e.promoted?'\u2605 ':'';
      row.innerHTML=`<div><b>${star}${esc(e.instruction)}</b></div>`
        +`<div style="opacity:.7;">${e.success}ok ${e.fail}x, ${(e.objects||[]).join(', ')}</div>`
        +`<button class="btn btn-s" onclick="togglePromote('${e.hash}',${!e.promoted})">`
        +`${e.promoted?'Demote':'Promote'}</button>`;
      list.appendChild(row);
    });
  }catch(e){
    document.getElementById('lib-list').textContent='error: '+e.message;
  }
}
async function togglePromote(h,promoted){
  try{
    await api('POST','/api/library/promote',{hash:h,promoted});
    showLibrary();
  }catch(e){toast('Promote failed: '+e.message,'err',4000)}
}

// Helpers
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
function toast(msg,cls='',ms=2500){
  const t=document.getElementById('toast');
  t.textContent=msg;t.className='toast '+cls+' show';
  if(t._toastTimer) clearTimeout(t._toastTimer);
  t._toastTimer=setTimeout(()=>t.className='toast',ms);
}

// Trial recorder
// Captures the live <img> feed into a hidden canvas at ~30 fps, pipes
// canvas.captureStream() into MediaRecorder, downloads a .webm on stop.
// Records whichever camera the viewer is currently showing (single, depth,
// or 'all' tile), whatever is in #cam-img.
let _rec={active:false,recorder:null,chunks:[],canvas:null,ctx:null,timer:null,stream:null,startedAt:0};

function toggleRecord(){ if(_rec.active) stopRecord(); else startRecord(); }

async function startRecord(){
  const img=document.getElementById('cam-img');
  if(!img||!img.naturalWidth){
    toast('Start a camera feed first','err');return;
  }
  if(typeof MediaRecorder==='undefined'){
    toast('MediaRecorder not supported in this browser','err');return;
  }
  const canvas=document.createElement('canvas');
  canvas.width=img.naturalWidth;
  canvas.height=img.naturalHeight;
  const ctx=canvas.getContext('2d');
  // Draw at ~30fps. Live feed updates slower (~12 fps); duplicates compress
  // to near-zero in webm so this is cheap.
  const timer=setInterval(()=>{
    if(!_rec.active) return;
    if(img.naturalWidth) ctx.drawImage(img,0,0,canvas.width,canvas.height);
  },1000/30);
  // Prime first frame
  ctx.drawImage(img,0,0,canvas.width,canvas.height);

  const stream=canvas.captureStream(30);
  let mime='video/webm;codecs=vp9';
  if(!MediaRecorder.isTypeSupported(mime)) mime='video/webm;codecs=vp8';
  if(!MediaRecorder.isTypeSupported(mime)) mime='video/webm';
  let recorder;
  try{
    recorder=new MediaRecorder(stream,{mimeType:mime,videoBitsPerSecond:4_000_000});
  }catch(e){
    clearInterval(timer);
    toast('Recorder init failed: '+e.message,'err');return;
  }
  const chunks=[];
  recorder.ondataavailable=e=>{ if(e.data&&e.data.size>0) chunks.push(e.data); };
  recorder.onstop=()=>{
    const blob=new Blob(chunks,{type:mime});
    const url=URL.createObjectURL(blob);
    const a=document.createElement('a');
    const ts=new Date().toISOString().replace(/[:.]/g,'-').slice(0,19);
    const dur=((Date.now()-_rec.startedAt)/1000).toFixed(1);
    a.href=url; a.download=`spark-trial-${ts}.webm`;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(()=>URL.revokeObjectURL(url),5000);
    log(`Recording saved: ${a.download} (${dur}s, ${(blob.size/1e6).toFixed(1)}MB)`,'ok');
    toast('Recording saved','ok',6000);
    if(stream) stream.getTracks().forEach(t=>t.stop());
  };
  recorder.start(1000); // chunk every 1s for memory safety on long trials
  _rec={active:true,recorder,chunks,canvas,ctx,timer,stream,startedAt:Date.now()};
  const btn=document.getElementById('btn-record');
  btn.classList.add('recording');
  btn.innerHTML='<span class="rec-dot"></span>Stop';
  log('Recording started','inf');
  toast('Recording...','inf');
}

function stopRecord(){
  if(!_rec.active) return;
  _rec.active=false;
  if(_rec.timer) clearInterval(_rec.timer);
  if(_rec.recorder&&_rec.recorder.state!=='inactive'){
    try{ _rec.recorder.stop(); }catch(e){}
  }
  const btn=document.getElementById('btn-record');
  btn.classList.remove('recording');
  btn.innerHTML='<span class="rec-dot"></span>Record';
}
function hlYaml(y){
  return (y||'')
    .replace(/^(\s*)([\w_]+):/gm,'$1<span class="k">$2</span>:')
    .replace(/:\s+"([^"]+)"/g,': <span class="s">"$1"</span>')
    .replace(/:\s+'([^']+)'/g,": <span class=\"s\">'$1'</span>")
    .replace(/:\s+(\d+\.?\d*)/g,': <span class="v">$1</span>');
}
