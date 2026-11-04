// Plan
async function doPlan(){
  const instr=document.getElementById('in-instr').value.trim();
  if(!instr){toast('Enter instruction','err');return}
  setStep('plan');log('Planning: "'+instr+'"','inf');
  try{
    const p=tags.length?tags:null;
    const d=await api('POST','/api/plan',{instruction:instr,prompts:p});
    document.getElementById('sec-plan').style.display='block';
    document.getElementById('plan-out').innerHTML=hlYaml(d.plan_yaml);
    if(d.detections) showDets(d.detections);
    log('Plan generated','ok');toast('Plan ready','ok');
  }catch(e){log('Plan failed: '+e.message,'err')}
}

// Execute with Approval
let _approvalOverlayMode = 'mask'; // 'mask', 'bbox', 'grasp'
let _approvalMerged = false; // false=raw detections, true=merged
let _approvalCamIdx = -1; // -1=scene cam, 0+=per-camera index
let _approvalData = null;

// Skip detection entirely and plan against state.pending_detections as it
// stands. This is the path for a hand-annotated scene: a click or a box has
// already put its detection in the pending set, and re-detecting would throw
// that work away.
async function doExecKeep(dry){
  const instr=document.getElementById('in-instr').value.trim();
  if(!instr){toast('Enter instruction','err');return}
  let cur;
  try{cur=await api('GET','/api/detections',null)}
  catch(e){log('Could not read detections: '+e.message,'err');return}
  const dets=(cur&&cur.detections)||[];
  if(!dets.length){
    log('Nothing detected yet -- click, box, or run Execute Pipeline first.','err');
    toast('No detections to reuse','err');return;
  }
  log('Reusing '+dets.length+' detection(s) without re-detecting:','inf');
  dets.forEach(d=>{
    const p=d.position_3d?`(${d.position_3d.map(v=>v.toFixed(3)).join(', ')})`:'N/A';
    log(`  ${d.label}: ${(d.confidence*100).toFixed(0)}% pos=${p}`,'inf');
  });
  // Approve first. Skipping detection is not a reason to skip the operator's
  // look at what is about to run -- and with click/box/text now accumulating,
  // the pending set can legitimately hold something detected several actions
  // ago that no longer belongs in the scene. The modal is where that gets seen
  // and discarded before the robot moves.
  _approvalData={instruction:instr,dry:dry,data:{
    detections:dets,
    prompts:dets.map(x=>x.label),
    raw_count:dets.length,
    merged_count:dets.length,
    mask_image:(_lastAnnotatedB64||''),
  }};
  _approvalCamIdx=-1;_approvalMerged=false;
  _showApprovalModal(_approvalData.data);
}

async function doExec(dry){
  const instr=document.getElementById('in-instr').value.trim();
  if(!instr){toast('Enter instruction','err');return}
  setStep('exec');log('"'+instr+'": detecting...','inf');
  toast('Detecting objects...','inf');

  // Step 1: Detect and show for approval
  try{
    const p=tags.length?tags:null;
    const useIr=document.getElementById('opt-detect-ir')?.checked||false;
    const d=await api('POST','/api/detect_approve',{instruction:instr,prompts:p,use_ir:useIr});
    _approvalData={instruction:instr,dry:dry,data:d};_approvalCamIdx=-1;_approvalMerged=false;
    log('Detected '+d.detections.length+' objects. Approve to continue.','ok');
    d.detections.forEach(det=>{
      const pos=det.position_3d?`(${det.position_3d[0].toFixed(3)}, ${det.position_3d[1].toFixed(3)}, ${det.position_3d[2].toFixed(3)})`:'N/A';
      log(`  ${det.label}: ${(det.confidence*100).toFixed(0)}% pos=${pos}`,'inf');
    });
    _showApprovalModal(d);
  }catch(e){
    log('Detection failed: '+e.message,'err');toast('Detection failed','err');
  }
}

function _showApprovalModal(d){
  // Remove existing modal if any
  let m=document.getElementById('approval-modal');
  if(m)m.remove();

  m=document.createElement('div');
  m.id='approval-modal';
  m.style.cssText='position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.85);z-index:9999;display:flex;flex-direction:column;align-items:center;padding:20px;overflow-y:auto';
  m.innerHTML=`
    <div style="color:#fff;font-size:18px;margin-bottom:10px">Detection Approval</div>
    <div style="display:flex;gap:8px;margin-bottom:6px">
      <button onclick="_setOverlay('mask')" id="btn-ov-mask" style="padding:6px 16px;border:2px solid #4af;background:#222;color:#fff;border-radius:4px;cursor:pointer;font-weight:bold">Masks</button>
      <button onclick="_setOverlay('bbox')" id="btn-ov-bbox" style="padding:6px 16px;border:2px solid #666;background:#222;color:#fff;border-radius:4px;cursor:pointer">Bboxes</button>
      <button onclick="_setOverlay('grasp')" id="btn-ov-grasp" style="padding:6px 16px;border:2px solid #666;background:#222;color:#fff;border-radius:4px;cursor:pointer">Grasp Preview</button>
      <span style="color:#666;margin:0 8px;line-height:36px">|</span>
      <button onclick="_toggleMerged()" id="btn-ov-merged" style="padding:6px 16px;border:2px solid #666;background:#222;color:#fff;border-radius:4px;cursor:pointer">Show Merged</button>
    </div>
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
      <button onclick="_prevCam()" style="padding:4px 12px;background:#333;color:#fff;border:1px solid #666;border-radius:4px;cursor:pointer;font-size:16px">&#9664;</button>
      <span style="color:#aaa;font-size:13px" id="cam-label">Scene (${d.raw_count} raw -> ${d.merged_count} merged)</span>
      <button onclick="_nextCam()" style="padding:4px 12px;background:#333;color:#fff;border:1px solid #666;border-radius:4px;cursor:pointer;font-size:16px">&#9654;</button>
    </div>
    <div style="position:relative;display:inline-block">
      <img id="approval-img" style="max-width:90vw;max-height:60vh;border:2px solid #444;border-radius:8px" src="data:image/jpeg;base64,${d.mask_image}">
      <div id="det-tooltip" style="position:fixed;display:none;background:#111;color:#fff;padding:8px 12px;border-radius:6px;border:1px solid #4af;font-size:13px;pointer-events:none;z-index:10001;white-space:nowrap;font-family:monospace"></div>
    </div>
    <div style="color:#aaa;font-size:13px;margin:10px 0">${d.detections.length} detections: ${d.prompts.join(', ')}</div>
    <!-- One row per detection with a discard control. A single bad detection
         used to force a full re-detect, which threw away every good one with
         it; dropping one is the cheap correction. -->
    <div id="det-rows" style="display:flex;flex-direction:column;gap:3px;margin-bottom:8px;max-height:22vh;overflow-y:auto;width:min(90vw,620px)"></div>
    <div style="display:flex;align-items:center;gap:8px;color:#aaa;font-size:13px;margin-bottom:4px">
      <span id="plan-source-tag" style="color:#666;font-family:monospace"></span>
      <!-- Read-only echo of the RF-DETR switch in the Actions panel: which
           second-opinion mode produced the confidences shown above. -->
      <span id="approval-proposer-tag" style="color:#666;font-family:monospace"></span>
    </div>
    <div style="display:flex;gap:12px;margin-top:10px;flex-wrap:wrap">
      <button onclick="_approveAndExecute()" style="padding:10px 30px;background:#2a2;color:#fff;border:none;border-radius:6px;font-size:16px;cursor:pointer">Approve & Execute</button>
      <button onclick="_approveAndExecute(true)" style="padding:10px 30px;background:#a80;color:#fff;border:none;border-radius:6px;font-size:16px;cursor:pointer">Dry Run</button>
      <button onclick="document.getElementById('approval-modal').remove()" style="padding:10px 30px;background:#a22;color:#fff;border:none;border-radius:6px;font-size:16px;cursor:pointer">Cancel</button>
    </div>
  `;
  document.body.appendChild(m);
  _renderDetRows();

  // BT cache pre-flight: will this task be served from the library, or will
  // it call the planner? Dry run only -- no planning, no detection, no robot.
  // Worth knowing BEFORE committing a scene reset to a run whose plan will
  // not be reproducible.
  (async()=>{
    const tag=document.getElementById('plan-source-tag');
    const instr=(_approvalData&&_approvalData.instruction)||'';
    if(!tag||!instr)return;
    try{
      const r=await(await fetch('/api/bt/resolve?instruction='+encodeURIComponent(instr))).json();
      const hit=(r.results||[])[0];
      if(!hit)return;
      if(hit.cached){
        tag.textContent='BT: '+hit.plan_source+' '+(hit.hash||'')+
          ' (success '+(hit.success||0)+'/fail '+(hit.fail||0)+')';
        tag.style.color='#4a6';
      }else{
        tag.textContent='BT: cache MISS - '+(hit.detail||'will call the planner');
        tag.style.color='#a80';
      }
    }catch(e){/* pre-flight only; never block approval on it */}
  })();

  // Which second-opinion mode the confidences above were fused under.
  (async()=>{
    const tag=document.getElementById('approval-proposer-tag');
    if(!tag)return;
    try{
      const s=await api('GET','/api/perception/fusion_status');
      if(s.available===false)return;
      tag.textContent='RF-DETR: '+(s.mode||'?')+
        ' ('+(s.fusion_enabled?(s.loaded_backend||s.backend||'none'):'gate off')+')';
      _renderProposerStatus(s);
    }catch(e){/* display only */}
  })();

  // Tooltip: hover over detections to see real-world position
  const _aimg=document.getElementById('approval-img');
  const _ttip=document.getElementById('det-tooltip');
  _aimg.addEventListener('mousemove',function(e){
    const rect=_aimg.getBoundingClientRect();
    const mx=e.clientX-rect.left, my=e.clientY-rect.top;
    const sx=_aimg.naturalWidth/rect.width, sy=_aimg.naturalHeight/rect.height;
    const ix=mx*sx, iy=my*sy;
    // Pick detection set based on current view
    const dets=_getTooltipDetections();
    let closest=null, minD=Infinity;
    for(const det of dets){
      if(!det.centroid_2d)continue;
      const dx=ix-det.centroid_2d[0], dy=iy-det.centroid_2d[1];
      const dist=Math.sqrt(dx*dx+dy*dy);
      if(dist<minD){minD=dist;closest=det}
    }
    if(closest&&minD<60){
      const p=closest.position_3d;
      const pos=p?`X:${p[0].toFixed(3)} Y:${p[1].toFixed(3)} Z:${p[2].toFixed(3)}`:'N/A';
      _ttip.innerHTML=`<b style="color:#4af">${closest.label}</b> (${(closest.confidence*100).toFixed(0)}%)<br>${pos}`;
      _ttip.style.display='block';
      _ttip.style.left=(e.clientX+14)+'px';
      _ttip.style.top=(e.clientY-10)+'px';
    }else{_ttip.style.display='none'}
  });
  _aimg.addEventListener('mouseleave',function(){_ttip.style.display='none'});
}

function _getTooltipDetections(){
  const d=_approvalData.data;
  const cams=d.cam_names||[];
  if(_approvalCamIdx===-1){
    return _approvalMerged?d.detections:(d.raw_detections||d.detections);
  }else{
    const camName=cams[_approvalCamIdx];
    const camData=(d.per_cam||{})[camName];
    return camData&&camData.detections?camData.detections:[];
  }
}

function _setOverlay(mode){
  _approvalOverlayMode=mode;
  _updateApprovalImg();
  ['mask','bbox','grasp'].forEach(m=>{
    const b=document.getElementById('btn-ov-'+m);
    if(b)b.style.borderColor=m===mode?'#4af':'#666';
  });
}

function _toggleMerged(){
  _approvalMerged=!_approvalMerged;
  const b=document.getElementById('btn-ov-merged');
  if(b){b.textContent=_approvalMerged?'Show Raw':'Show Merged';b.style.borderColor=_approvalMerged?'#4af':'#666'}
  _updateApprovalImg();
}

function _prevCam(){
  const d=_approvalData.data;
  const cams=d.cam_names||[];
  // -1=scene, 0..n=per-camera
  _approvalCamIdx--;
  if(_approvalCamIdx<-1)_approvalCamIdx=cams.length-1;
  _updateApprovalImg();
}
function _nextCam(){
  const d=_approvalData.data;
  const cams=d.cam_names||[];
  _approvalCamIdx++;
  if(_approvalCamIdx>=cams.length)_approvalCamIdx=-1;
  _updateApprovalImg();
}

function _updateApprovalImg(){
  const d=_approvalData.data;
  const img=document.getElementById('approval-img');
  const label=document.getElementById('cam-label');
  const m=_approvalOverlayMode;
  const cams=d.cam_names||[];

  if(_approvalCamIdx===-1){
    // Scene view (raw or merged)
    const key=_approvalMerged?m+'_merged':m+'_image';
    img.src='data:image/jpeg;base64,'+(d[key]||d[m+'_image']);
    label.textContent=(_approvalMerged?'Merged':'Scene')+' ('+d.raw_count+' raw -> '+d.merged_count+' merged)';
  }else{
    // Per-camera view
    const camName=cams[_approvalCamIdx];
    const camData=(d.per_cam||{})[camName];
    if(camData){
      img.src='data:image/jpeg;base64,'+camData[m];
      label.textContent=camName+' ('+camData.count+' detections)';
    }else{
      label.textContent=camName+' (no data)';
    }
  }
}
