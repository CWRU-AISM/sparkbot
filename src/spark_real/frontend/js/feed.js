// Global keys (teleop works globally, not just in zone)
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA') return;

  // Teleop keys work globally (no need to focus zone)
  const k=e.key.toLowerCase();
  if(k==='escape'){doStop();e.preventDefault();return}
  if(k==='h'){doHome();e.preventDefault();return}
  if(k==='g'){
    gripperOpen=!gripperOpen;
    grip(gripperOpen?'open':'close');
    e.preventDefault();return;
  }
  if(TP.keys[k]){
    if(!teleopActive){
      teleopActive=true;
      document.getElementById('teleop-status').innerHTML='<span class="live-dot"></span>Teleop active';
      if(!teleopInterval) teleopInterval=setInterval(teleopTick,33);
    }
    if(!heldKeys.has(k)){
      heldKeys.add(k);
      updateKeyIndicators();
    }
    e.preventDefault();return;
  }

  if(e.target.id==='teleop-zone') return;
  if(e.key==='1')openTab('task');if(e.key==='2')openTab('robot');
  if(e.key==='3')openTab('log');if(e.key==='4')openTab('hist');
  if(e.key==='c'&&!e.ctrlKey)doCapture();
  if(e.key==='d')doDetect();
  if(e.key==='Escape')doStop();
});
document.addEventListener('keyup',e=>{
  const k=e.key.toLowerCase();
  if(heldKeys.has(k)){
    heldKeys.delete(k);
    updateKeyIndicators();
    // If no keys held, stop after brief delay
    if(heldKeys.size===0 && teleopActive){
      sendStopVelocity();
    }
  }
});

// Live camera feed (polling /api/capture/fast)
let liveFeedRunning=false;
let liveFps=0;
let liveFrameCount=0;
let liveFpsTimer=null;
let _depthMode=false;
let _irMode=false;

async function _applyStreamMode(){
  // depth and ir are mutually exclusive with rgb. depth takes precedence
  // if both flags somehow get set (shouldn't happen via the buttons).
  const mode = _depthMode ? 'depth' : (_irMode ? 'ir' : 'rgb');
  const btnD=document.getElementById('btn-depth');
  const btnI=document.getElementById('btn-ir');
  btnD.style.background=_depthMode?'var(--cyan)':'';
  btnD.style.color=_depthMode?'#000':'';
  btnI.style.background=_irMode?'var(--cyan)':'';
  btnI.style.color=_irMode?'#000':'';
  await api('POST','/api/stream_mode',{mode});
  const tag = mode === 'depth' ? 'DEPTH (turbo)'
            : mode === 'ir'    ? 'IR (active)'
            : 'LIVE';
  document.getElementById('cam-tag').textContent=tag;
  // If a detection overlay is pinned, the static annotated image is
  // currently showing. Switch back to live feed (in the newly-selected
  // mode) so the persistent bbox overlay renders on top of the live
  // stream the operator just asked for.
  if(_persistentOverlay && _showingDetection){
    _showingDetection=false;
    if(!liveFeedRunning) startLiveFeed();
  }
}

async function toggleDepth(){
  _depthMode=!_depthMode;
  if(_depthMode) _irMode=false;
  await _applyStreamMode();
}

async function toggleIR(){
  _irMode=!_irMode;
  if(_irMode) _depthMode=false;
  await _applyStreamMode();
}


let _streamCam='all';
let _zoomLevel=1, _panX=0, _panY=0, _isPanning=false, _panStart={x:0,y:0};
async function switchStreamCam(cam){
  _streamCam=cam;
  await api('POST','/api/stream_camera',{camera:cam});
  const tag=document.getElementById('cam-tag');
  tag.textContent=cam==='all'?'LIVE':cam.toUpperCase();
  // Reset zoom when switching
  _zoomLevel=1; _panX=0; _panY=0;
  _applyZoom();
}

function _applyZoom(){
  const img=document.getElementById('cam-img');
  if(_streamCam==='all' || _zoomLevel<=1){
    img.style.transform='';
    img.style.transformOrigin='';
    img.style.cursor='';
    _zoomLevel=1; _panX=0; _panY=0;
  } else {
    img.style.transform=`scale(${_zoomLevel}) translate(${_panX}px, ${_panY}px)`;
    img.style.transformOrigin='center center';
    img.style.cursor='grab';
  }
}

function _setupZoom(){
  const vp=document.getElementById('viewport');
  vp.addEventListener('wheel',e=>{
    if(_streamCam==='all') return;
    e.preventDefault();
    const delta=e.deltaY>0?-0.2:0.2;
    _zoomLevel=Math.max(1, Math.min(8, _zoomLevel+delta));
    if(_zoomLevel<=1){_panX=0;_panY=0;}
    _applyZoom();
  },{passive:false});

  vp.addEventListener('mousedown',e=>{
    if(_streamCam==='all'||_zoomLevel<=1) return;
    if(e.button!==1 && !e.shiftKey) return; // middle-click or shift+click to pan
    _isPanning=true;
    _panStart={x:e.clientX-_panX, y:e.clientY-_panY};
    e.preventDefault();
  });
  window.addEventListener('mousemove',e=>{
    if(!_isPanning) return;
    _panX=e.clientX-_panStart.x;
    _panY=e.clientY-_panStart.y;
    _applyZoom();
  });
  window.addEventListener('mouseup',()=>{_isPanning=false;});
}

// Sticky pause: survives the many auto-restart call sites (post-detect,
// post-execute, cam switch). While paused the server encodes zero frames
// for this tab; prompts, teleop, and status keep working.
let _feedPaused=false;
function toggleFeed(){
  _feedPaused=!_feedPaused;
  const b=document.getElementById('btn-feed');
  if(_feedPaused){stopLiveFeed();if(b)b.textContent='Feed: off';}
  else{if(b)b.textContent='Feed: on';startLiveFeed();}
}
function startLiveFeed(){
  if(liveFeedRunning||_feedPaused) return;
  liveFeedRunning=true;
  liveFpsTimer=setInterval(()=>{liveFps=liveFrameCount;liveFrameCount=0},1000);

  async function tick(){
    if(!liveFeedRunning || _showingDetection) return;
    try{
      const res=await fetch(API+'/api/capture/stream');
      if(!res.ok) throw new Error(res.statusText);
      // Check again after await, detection might have started during fetch
      if(!liveFeedRunning || _showingDetection) return;
      const blob=await res.blob();
      const url=URL.createObjectURL(blob);
      const img=document.getElementById('cam-img');
      const old=img._blobUrl;
      img.onload=()=>{
        if(old) URL.revokeObjectURL(old);
        // Detection bboxes survive mode/cam switches because they're
        // redrawn on the canvas overlay every frame.
        if(_persistentOverlay) redrawPersistentOverlay();
      };
      img._blobUrl=url;
      img.src=url;
      img.style.display='block';
      document.getElementById('cam-empty').style.display='none';
      document.getElementById('cam-tag').style.display='block';
      document.getElementById('cam-tag').textContent=_depthMode?'DEPTH (turbo)':((_streamCam&&_streamCam!=='all')?_streamCam:'live').toUpperCase();
      document.getElementById('cam-ts').style.display='block';
      document.getElementById('cam-ts').textContent=liveFps+'fps';
      liveFrameCount++;
    }catch(e){}
    if(liveFeedRunning && !_showingDetection) setTimeout(tick, 80);
  }
  tick();
}

function stopLiveFeed(){
  liveFeedRunning=false;
  if(liveFpsTimer){clearInterval(liveFpsTimer);liveFpsTimer=null}
}
