// Detect (text)
async function doDetect(){
  const p=tags.length?tags:autoExtract();
  if(!p.length){toast('Add prompts first','err');return}
  stopLiveFeed();
  setStep('det');log('Detecting: '+p.join(', '),'inf');
  document.getElementById('cam-tag').textContent='DETECTING...';
  try{
    const multiInst=document.getElementById('multi-inst')?.checked||false;
    const d=await api('POST','/api/detect',{prompts:p,multi_instance:multiInst});
    if(d.annotated_image){
      // Keep detection masks visible until user clicks image or clicks Execute
      stopLiveFeed();
      _showingDetection=true;
      const img=document.getElementById('cam-img');
      img.src='data:image/jpeg;base64,'+d.annotated_image;
      img.style.display='block';
      document.getElementById('cam-empty').style.display='none';
      document.getElementById('cam-tag').textContent='DETECTIONS (click to resume)';
      _lastAnnotatedB64=d.annotated_image;
      document.getElementById('cam-tag').style.display='block';
      console.log('Detection image set, size='+d.annotated_image.length);
    } else {
      console.log('No annotated_image in response');
      startLiveFeed();
    }
    await refreshDets(d.detections);
    log('Found '+d.count+' objects','ok');
    d.detections.forEach(o=>{
      const pos=o.position_3d?`(${o.position_3d.map(v=>v.toFixed(3)).join(', ')})`:'N/A';
      log('  '+o.label+': '+Math.round(o.confidence*100)+'% pos='+pos,'acc');
    });
  }catch(e){
    log('Detection failed: '+e.message,'err');
    _showingDetection=false;
    startLiveFeed();
  }
}

// Detect (click)
async function doClickDetect(x,y,w,h){
  const label=prompt('Label this object (or leave blank):','')||'';
  setStep('det');log(`Click detect at (${Math.round(x)},${Math.round(y)})`,'inf');
  try{
    const d=await api('POST','/api/detect_click',{x,y,img_width:w,img_height:h,label});
    if(d.annotated_image){_lastAnnotatedB64=d.annotated_image;showImage('data:image/jpeg;base64,'+d.annotated_image);}
    await refreshDets(d.detections);
    log('Click detected: '+(d.detections[0]?.label||'nothing')+' ('+Math.round((d.detections[0]?.confidence||0)*100)+'%)','ok');
  }catch(e){log('Click detect failed: '+e.message,'err')}
}

// Detect (box)
async function doBoxDetect(x1,y1,x2,y2,w,h){
  const label=prompt('Label this object (or leave blank):','')||'';
  setStep('det');log(`Box detect (${Math.round(x1)},${Math.round(y1)})-(${Math.round(x2)},${Math.round(y2)})`,'inf');
  try{
    const d=await api('POST','/api/detect_box',{x1,y1,x2,y2,img_width:w,img_height:h,label});
    if(d.annotated_image){_lastAnnotatedB64=d.annotated_image;showImage('data:image/jpeg;base64,'+d.annotated_image);}
    await refreshDets(d.detections);
    log('Box detected: '+(d.detections[0]?.label||'nothing'),'ok');
  }catch(e){log('Box detect failed: '+e.message,'err')}
}

// Show detections
const DET_COLORS=['#ff3c3c','#3cdc78','#3c78ff','#ffc828','#dc3cdc','#3cdcdc','#ff8c28','#8cff28','#288cff'];
let _currentDets=[];
let _persistentOverlay=false;  // true: draw bboxes over every live frame

// Render the server's pending set rather than the response of whichever
// detect just ran. The response is deliberately per-call -- a text detect
// reports what IT found and must not claim credit for earlier annotations --
// but the strip is meant to show everything that will actually execute. Using
// the response made every text detect look like it had wiped the click and box
// work, when the server had kept it all along.
// Last annotated frame any detect returned, so the approval modal has a
// preview to show when a run reuses detections instead of making new ones.
let _lastAnnotatedB64='';

async function refreshDets(fallback){
  try{
    const cur=await api('GET','/api/detections',null);
    const all=(cur&&cur.detections)||[];
    showDets(all);
    return all;
  }catch(e){
    if(fallback) showDets(fallback);
    return fallback||[];
  }
}

function showDets(dets){
  _currentDets=dets;
  // Persistent overlay: keep masks visible on every live frame so the
  // operator can switch RGB/IR/Depth without losing the detections.
  // Cleared by clicking the image or running a new detection.
  _persistentOverlay = (dets && dets.length > 0);
  const strip=document.getElementById('det-strip');
  const items=document.getElementById('det-items');
  const count=document.getElementById('det-count');
  strip.style.display='block';count.textContent=dets.length;items.innerHTML='';
  dets.forEach((d,i)=>{
    const pos=d.position_3d?d.position_3d.map(v=>v.toFixed(2)).join(','):'';
    const el=document.createElement('div');el.className='det-chip fade';
    el.style.animationDelay=i*40+'ms';
    el.innerHTML=`<span class="det-swatch" style="background:${DET_COLORS[i%9]}"></span>`
      +`<span class="det-name">${esc(d.label)}</span>`
      +`<span class="det-conf">${Math.round(d.confidence*100)}%</span>`
      +(d.camera?`<span class="det-pos">${d.camera}</span>`:'')
      +(pos?`<span class="det-pos">[${pos}]</span>`:'')
      // Discard control on the chip itself. One bad detection used to force a
      // full re-detect, which threw away every good one along with it -- and
      // now that click/box/text accumulate, pruning is the other half of that.
      +`<span class="det-x" title="Discard this detection" role="button">&#128465;</span>`;
    el.addEventListener('mouseenter',()=>highlightDet(i));
    el.addEventListener('mouseleave',clearHighlight);
    const x=el.querySelector('.det-x');
    if(x) x.addEventListener('click',ev=>{ev.stopPropagation();dropDetChip(d.label);});
    items.appendChild(el);
  });
}

// Drop one detection from the server's pending set, then re-render from what
// the server says is actually left rather than trusting the local edit.
async function dropDetChip(label){
  try{
    await api('POST','/api/detections/drop',{labels:[label]});
    const cur=await api('GET','/api/detections',null);
    const left=(cur&&cur.detections)||[];
    log('Discarded detection: '+label+' ('+left.length+' left)','inf');
    if(left.length) showDets(left);
    else{
      _currentDets=[];_persistentOverlay=false;
      const strip=document.getElementById('det-strip');
      const items=document.getElementById('det-items');
      const count=document.getElementById('det-count');
      if(items)items.innerHTML='';
      if(count)count.textContent='0';
      if(strip)strip.style.display='none';
    }
  }catch(e){log('Could not discard '+label+': '+e.message,'err')}
}

function highlightDet(idx){
  const d=_currentDets[idx];
  if(!d) return;
  // Pause live feed while hovering so overlay is visible
  if(liveFeedRunning) stopLiveFeed();
  const img=document.getElementById('cam-img');
  const canvas=document.getElementById('overlay-canvas');
  if(!img||!canvas||img.style.display==='none') return;

  const rect=img.getBoundingClientRect();
  const natW=img.naturalWidth;
  const natH=img.naturalHeight;
  if(!natW||!natH) return;

  // object-fit:contain, compute actual rendered area
  const imgAspect=natW/natH;
  const boxAspect=rect.width/rect.height;
  let rw,rh,ox,oy;
  if(imgAspect>boxAspect){
    rw=rect.width; rh=rect.width/imgAspect;
    ox=0; oy=(rect.height-rh)/2;
  } else {
    rh=rect.height; rw=rect.height*imgAspect;
    oy=0; ox=(rect.width-rw)/2;
  }

  canvas.width=rect.width;
  canvas.height=rect.height;
  canvas.style.display='block';
  canvas.style.pointerEvents='none';
  const ctx=canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);

  const color=DET_COLORS[idx%9];

  // The annotated image is tiled (side|bird|wrist). We need to draw a
  // pulsing highlight around the detection's region in its source camera tile.
  // For now, draw a large label + crosshair at the detection's centroid
  // mapped through the tile layout.

  // Use tiled_bbox (already in tiled image pixel coords) for hover
  const tb=d.tiled_bbox;
  if(tb){
    const [x1,y1,x2,y2]=tb;
    const sx=rw/natW, sy=rh/natH;
    const bx=ox+x1*sx, by=oy+y1*sy, bw=(x2-x1)*sx, bh=(y2-y1)*sy;

    // Glow highlight
    ctx.shadowColor=color;
    ctx.shadowBlur=12;
    ctx.strokeStyle=color;
    ctx.lineWidth=3;
    ctx.strokeRect(bx,by,bw,bh);
    ctx.shadowBlur=0;

    // Label
    ctx.font='bold 14px monospace';
    const label=d.label+' '+Math.round(d.confidence*100)+'%'+(d.camera?' ('+d.camera+')':'');
    const tw=ctx.measureText(label).width;
    ctx.fillStyle=color;
    ctx.fillRect(bx,Math.max(0,by-20),tw+10,20);
    ctx.fillStyle='#000';
    ctx.fillText(label,bx+5,Math.max(14,by-5));

    // Crosshair at center
    const ccx=bx+bw/2, ccy=by+bh/2;
    ctx.strokeStyle=color;
    ctx.lineWidth=2;
    ctx.beginPath();ctx.moveTo(ccx-12,ccy);ctx.lineTo(ccx+12,ccy);ctx.stroke();
    ctx.beginPath();ctx.moveTo(ccx,ccy-12);ctx.lineTo(ccx,ccy+12);ctx.stroke();
    ctx.beginPath();ctx.arc(ccx,ccy,8,0,Math.PI*2);ctx.stroke();
  }
}

function redrawPersistentOverlay(){
  // Redraw all current detections as bbox + label overlays on top of
  // whatever live frame is showing. Survives mode switches because the
  // detection data is kept client-side. Called from the live-feed tick
  // whenever a new frame paints.
  if(!_persistentOverlay || !_currentDets || _currentDets.length === 0) return;
  const img=document.getElementById('cam-img');
  const canvas=document.getElementById('overlay-canvas');
  if(!img || !canvas || img.style.display === 'none') return;
  const rect=img.getBoundingClientRect();
  const natW=img.naturalWidth, natH=img.naturalHeight;
  if(!natW || !natH) return;
  const imgAspect=natW/natH;
  const boxAspect=rect.width/rect.height;
  let rw,rh,ox,oy;
  if(imgAspect>boxAspect){
    rw=rect.width; rh=rect.width/imgAspect;
    ox=0; oy=(rect.height-rh)/2;
  } else {
    rh=rect.height; rw=rect.height*imgAspect;
    oy=0; ox=(rect.width-rw)/2;
  }
  canvas.width=rect.width; canvas.height=rect.height;
  canvas.style.display='block';
  canvas.style.pointerEvents='none';
  const ctx=canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  const sx=rw/natW, sy=rh/natH;
  _currentDets.forEach((d, idx) => {
    const tb=d.tiled_bbox;
    if(!tb) return;
    const [x1,y1,x2,y2]=tb;
    const bx=ox+x1*sx, by=oy+y1*sy, bw=(x2-x1)*sx, bh=(y2-y1)*sy;
    const color=DET_COLORS[idx%9];
    ctx.shadowColor=color; ctx.shadowBlur=8;
    ctx.strokeStyle=color; ctx.lineWidth=2;
    ctx.strokeRect(bx,by,bw,bh);
    ctx.shadowBlur=0;
    const label=d.label+' '+Math.round(d.confidence*100)+'%';
    ctx.font='bold 12px monospace';
    const tw=ctx.measureText(label).width;
    ctx.fillStyle=color;
    ctx.fillRect(bx,Math.max(0,by-18),tw+8,18);
    ctx.fillStyle='#000';
    ctx.fillText(label,bx+4,Math.max(13,by-5));
  });
}

function clearPersistentOverlay(){
  _persistentOverlay=false;
  const canvas=document.getElementById('overlay-canvas');
  if(canvas){
    const ctx=canvas.getContext('2d');
    ctx.clearRect(0,0,canvas.width,canvas.height);
    canvas.style.display='none';
  }
}

function clearHighlight(){
  const canvas=document.getElementById('overlay-canvas');
  if(!canvas) return;
  const ctx=canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  canvas.style.display='none';
  // Resume live feed if not showing detection
  if(!_showingDetection && !liveFeedRunning) startLiveFeed();
}
