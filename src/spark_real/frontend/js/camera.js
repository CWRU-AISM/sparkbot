// Capture
async function doCapture(){
  setStep('cap');log('Capturing...','inf');
  try{
    const d=await api('GET','/api/capture');
    const sel=(_streamCam&&_streamCam!=='all')?_streamCam:'sideview';
    const cam=d[sel]||d.sideview||Object.values(d)[0];
    if(!cam) return;
    const src=sel==='depth'&&cam.depth?cam.depth:cam.rgb;
    if(!src){log('No data for '+sel,'err');return}
    showImage('data:image/jpeg;base64,'+src, sel.toUpperCase());
    log('Captured '+cam.width+'x'+cam.height,'ok');
  }catch(e){log('Capture failed: '+e.message,'err')}
}

let _showingDetection=false;
let _flashTimer=null;
function flashImage(src,label,durationMs){
  // Show an image briefly then auto-resume live feed
  if(_flashTimer){clearTimeout(_flashTimer);_flashTimer=null}
  const img=document.getElementById('cam-img');
  img.src=src;img.style.display='block';
  document.getElementById('cam-empty').style.display='none';
  document.getElementById('cam-tag').textContent=label||'';
  document.getElementById('cam-tag').style.display='block';
  const ts=new Date().toLocaleTimeString('en',{hour12:false});
  document.getElementById('cam-ts').textContent=ts;
  document.getElementById('cam-ts').style.display='block';
  _showingDetection=true;
  stopLiveFeed();
  // durationMs<=0 means HOLD: the frame stays until the operator asks for
  // live again. A detection you cannot look at is a detection you cannot
  // check, and 3 s was not enough to read a mask against the scene.
  if(durationMs>0){
    _flashTimer=setTimeout(()=>{
      _showingDetection=false;_flashTimer=null;
      startLiveFeed();
      document.getElementById('cam-tag').textContent='LIVE';
    },durationMs);
  }else{
    const tag=document.getElementById('cam-tag');
    if(tag) tag.textContent=(label||'DETECTION')+'  -  click to resume live';
  }
}
// Click the frame to drop the held detection and go back to live.
function resumeLive(){
  if(_flashTimer){clearTimeout(_flashTimer);_flashTimer=null}
  _showingDetection=false;
  startLiveFeed();
  const tag=document.getElementById('cam-tag');
  if(tag) tag.textContent='LIVE';
}
// HOLD by default. Callers that genuinely want a flash pass a duration.
function showImage(src,label,durationMs){flashImage(src,label,durationMs===undefined?0:durationMs)}
// Click on image to resume immediately (skip the timer)
document.addEventListener('DOMContentLoaded',()=>{
  const img=document.getElementById('cam-img');
  if(img) img.addEventListener('click',e=>{
    if(_showingDetection){
      if(_flashTimer){clearTimeout(_flashTimer);_flashTimer=null}
      _showingDetection=false;
      clearPersistentOverlay();   // dismiss bbox overlay too
      startLiveFeed();
      document.getElementById('cam-tag').textContent='LIVE';
    }
  });
});

// RF-DETR second opinion (perception.fusion.proposer)
// The select is a runtime override of the family YAML: the server drops its
// cached fusion gate on every change, so the new mode is live on the next
// detect with no restart. The status line beside it is READ-ONLY -- whether
// the fusion gate as a whole is up, and which backend is actually loaded.
function _renderProposerStatus(s){
  const sel=document.getElementById('opt-proposer-mode');
  const tag=document.getElementById('proposer-status');
  if(!s||s.available===false){if(tag)tag.textContent='';return}
  if(sel&&s.mode)sel.value=s.mode;               // CURRENT mode, not a guess
  if(!tag)return;
  const backend=s.loaded_backend||s.backend||'none';
  const src=s.source&&s.source!=='config'?' via '+s.source:'';
  tag.textContent='gate '+(s.fusion_enabled?'on':'off')+' / '+backend+src;
  tag.style.color=s.error?'#c55':(s.fusion_enabled?'#666':'#a80');
  tag.title=s.error?s.error:
    'fusion gate: '+(s.gate_state||'?')+'\nbackend: '+backend+
    '\nscored labels: '+((s.labels&&s.labels.length)?s.labels.join(', '):'(all in-vocabulary)');
}
async function refreshProposerMode(){
  try{_renderProposerStatus(await api('GET','/api/perception/fusion_status'))}
  catch(e){/* status only; never block the detect panel on it */}
}
async function setProposerMode(mode){
  try{
    const s=await api('POST','/api/perception/proposer_mode',{mode:mode});
    _renderProposerStatus(s);
    log('RF-DETR second opinion: '+mode,'ok');
  }catch(e){
    log('RF-DETR mode change failed: '+e.message,'err');
    refreshProposerMode();   // snap the select back to the server's truth
  }
}
document.addEventListener('DOMContentLoaded',refreshProposerMode);

// Camera switch
