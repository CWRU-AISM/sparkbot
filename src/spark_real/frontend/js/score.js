// BT Library panel
// Cached behaviour-trees from prior successful runs. Operator can:
//   - flip the "Use Gemini / Use Cached BT" toggle (runtime override
//     for SPARK_DISABLE_LLM; UI > env > default OFF).
//   - pin a specific BT for the NEXT /api/execute (single-shot,
//     auto-clears server-side once pipeline.plan() consumes it).
//   - promote (star) entries so the planner always sees them as few-shot
//     examples regardless of jaccard similarity to the new instruction.
//   - delete entries that were bad.
// Polls /api/bt/list + /api/bt/cache_status every 10s; re-renders only
// when the content hash changes so the scroll position is preserved.
let _btLibEntries = [];
let _btLibStatus = {enabled:false,source:'default'};
let _btLibPinned = null;
let _btLibExpanded = new Set();
let _btLibCollapsed = false;
let _btLibLastHash = '';
function _btLibToggleCollapse(){
  _btLibCollapsed = !_btLibCollapsed;
  const body = document.getElementById('bt-lib-body');
  const ind = document.getElementById('bt-lib-collapse-ind');
  if(body) body.style.display = _btLibCollapsed ? 'none' : '';
  if(ind) ind.textContent = _btLibCollapsed ? '\u25b8' : '\u25be';
}
function _btLibHashContent(){
  // Cheap content hash so we skip re-renders when nothing changed.
  return JSON.stringify({
    n: _btLibEntries.length,
    e: _btLibStatus,
    p: _btLibPinned,
    h: _btLibEntries.map(e => e.hash + '|' + (e.promoted?1:0) + '|' + (e.success||0))
  });
}
async function _btLibRefresh(){
  try{
    const [list, status] = await Promise.all([
      api('GET','/api/bt/list'),
      api('GET','/api/bt/cache_status'),
    ]);
    _btLibEntries = (list && list.entries) || [];
    _btLibPinned = (list && list.pinned) || null;
    _btLibStatus = status || {enabled:false,source:'default'};
    const h = _btLibHashContent();
    if(h !== _btLibLastHash){
      _btLibLastHash = h;
      _btLibRender();
    }
  }catch(e){
    // Server may be mid-boot; stay quiet on transient errors.
  }
}
function _btLibRender(){
  // Toggle button reflects the resolved state (UI > env > default).
  const btn = document.getElementById('bt-lib-toggle-btn');
  const src = document.getElementById('bt-lib-source');
  const count = document.getElementById('bt-lib-count');
  if(count) count.textContent = '(' + _btLibEntries.length + ')';
  if(btn){
    if(_btLibStatus.enabled){
      btn.textContent = 'ok Using Cached BT';
      btn.style.background = 'var(--cyan-15)';
      btn.style.borderColor = 'var(--cyan-30)';
      btn.style.color = 'var(--cyan)';
    }else{
      btn.textContent = 'Using Gemini (LLM)';
      btn.style.background = '';
      btn.style.borderColor = '';
      btn.style.color = '';
    }
  }
  if(src){
    src.textContent = 'source: ' + (_btLibStatus.source || '?');
  }
  // Pin banner
  const pinBanner = document.getElementById('bt-lib-pin-banner');
  const pinText = document.getElementById('bt-lib-pin-text');
  if(_btLibPinned){
    const e = _btLibEntries.find(x => x.hash === _btLibPinned);
    if(pinBanner) pinBanner.style.display = '';
    if(pinText){
      const lbl = e ? e.instruction : _btLibPinned;
      pinText.textContent = 'pinned: ' + (lbl.length > 50 ? lbl.slice(0,50)+'...' : lbl);
    }
  }else{
    if(pinBanner) pinBanner.style.display = 'none';
  }
  // List
  const list = document.getElementById('bt-lib-list');
  const q = (document.getElementById('bt-lib-search')||{}).value || '';
  const qLower = q.trim().toLowerCase();
  const filtered = qLower
    ? _btLibEntries.filter(e => (e.instruction||'').toLowerCase().includes(qLower))
    : _btLibEntries;
  if(!list) return;
  if(!filtered.length){
    list.innerHTML = '<div style="padding:8px;color:var(--tx-3);font:400 10px var(--mono);text-align:center">' + (_btLibEntries.length ? 'no matches' : 'library is empty') + '</div>';
    return;
  }
  const rows = filtered.map(e => {
    const isPinned = e.hash === _btLibPinned;
    const instr = e.instruction || '(no instruction)';
    const instrTrunc = instr.length > 60 ? instr.slice(0,60)+'...' : instr;
    const expanded = _btLibExpanded.has(e.hash);
    const starChar = e.promoted ? '\u2605' : '\u2606';
    const starColor = e.promoted ? '#ffcc40' : 'var(--tx-3)';
    return (
      '<div class="bt-lib-row" data-hash="'+e.hash+'" style="border-bottom:1px solid var(--border);padding:6px 8px;'+(isPinned?'background:var(--bg-3);':'')+'">' +
      '  <div style="display:flex;align-items:center;gap:6px">' +
      '    <span style="cursor:pointer;color:'+starColor+';font-size:13px;flex:0 0 auto" onclick="_btLibPromote(event,\''+e.hash+'\','+(!e.promoted)+')" title="'+(e.promoted?'Unpromote':'Promote')+' (always-include in planner prompts)">'+starChar+'</span>' +
      '    <span style="flex:1;cursor:pointer;font:500 11px var(--mono);color:var(--tx-1);overflow:hidden;text-overflow:ellipsis;white-space:nowrap" onclick="_btLibExpand(\''+e.hash+'\')" title="'+_btEsc(instr)+'">'+_btEsc(instrTrunc)+'</span>' +
      '    <span style="flex:0 0 auto;font:400 9px var(--mono);color:var(--tx-3)">'+e.n_actions+' act</span>' +
      '    <button class="btn btn-s" style="flex:0 0 auto;padding:2px 6px;font-size:9px;'+(isPinned?'background:var(--cyan-15);color:var(--cyan)':'')+'" onclick="_btLibPin(event,\''+e.hash+'\')" title="Use this BT for the NEXT execute call (single-shot)">'+(isPinned?'Pinned':'Pin')+'</button>' +
      '    <button class="btn btn-s" style="flex:0 0 auto;padding:2px 6px;font-size:9px;background:#5a2020;color:#ffaaaa" onclick="_btLibDelete(event,\''+e.hash+'\')" title="Delete this BT">Del</button>' +
      '  </div>' +
      (expanded ? _btLibRenderDetail(e) : '') +
      '</div>'
    );
  });
  list.innerHTML = rows.join('');
}
function _btEsc(s){
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,"&#39;");
}
function _btLibRenderDetail(e){
  // Lazy fetch on first expand: list response only has summary fields.
  // The detail block is rendered from the cached score we fetch in
  // _btLibExpand and store on the entry object.
  if(!e._detail) return '<div style="padding:4px 0 4px 22px;font:400 9px var(--mono);color:var(--tx-3)">loading detail...</div>';
  const d = e._detail;
  const tree = (d.score && d.score.tree) || {};
  const labels = [];
  const actionTypes = [];
  (function walk(node){
    if(!node || typeof node !== 'object') return;
    const p = node.params || {};
    if(p.keypoint_label && !labels.includes(p.keypoint_label)) labels.push(p.keypoint_label);
    if(node.type && !['sequence','selector'].includes(node.type)) actionTypes.push(node.type);
    (node.children||[]).forEach(walk);
  })(tree);
  // Predicate labels sit outside the tree; the panel must list them too, or a
  // tree looks like it needs fewer detections than it actually does.
  (((d.score && d.score.verify && d.score.verify.all) || [])).forEach(function(pr){
    if(!pr || typeof pr !== 'object') return;
    ['obj','container','surface','base','target'].forEach(function(k){
      if(typeof pr[k] === 'string' && pr[k] && !labels.includes(pr[k])) labels.push(pr[k]);
    });
  });
  const created = d.created_at ? d.created_at.replace('T',' ').slice(0,19) : '?';
  return (
    '<div style="padding:4px 0 6px 22px;font:400 9px var(--mono);color:var(--tx-2);line-height:1.5">' +
    '  <div><span style="color:var(--tx-3)">hash:</span> '+_btEsc(e.hash)+' <span style="color:var(--tx-3)">created:</span> '+_btEsc(created)+'</div>' +
    '  <div><span style="color:var(--tx-3)">labels:</span> '+_btEsc(labels.join(', ') || '(none)')+'</div>' +
    '  <div><span style="color:var(--tx-3)">actions ('+actionTypes.length+'):</span> '+_btEsc(actionTypes.join(' -> ') || '(none)')+'</div>' +
    '  <div><span style="color:var(--tx-3)">success:</span> '+(d.success||0)+' <span style="color:var(--tx-3)">fail:</span> '+(d.fail||0)+'</div>' +
    '</div>'
  );
}
async function _btLibExpand(hash){
  if(_btLibExpanded.has(hash)){
    _btLibExpanded.delete(hash);
    _btLibRender();
    return;
  }
  _btLibExpanded.add(hash);
  _btLibRender();
  // Fetch detail if not cached.
  const e = _btLibEntries.find(x => x.hash === hash);
  if(e && !e._detail){
    try{
      const d = await api('GET','/api/bt/get?hash=' + encodeURIComponent(hash));
      if(d && !d.error){
        e._detail = d;
        _btLibRender();
      }
    }catch(_){}
  }
}
async function _btLibToggleCache(){
  const newVal = !_btLibStatus.enabled;
  try{
    const d = await api('POST','/api/bt/toggle_cache',{enabled:newVal});
    if(d && typeof d.enabled === 'boolean'){
      _btLibStatus = d;
      _btLibLastHash = '';
      _btLibRender();
      log('BT library: ' + (d.enabled ? 'using cached BTs' : 'using Gemini') + ' (source=' + d.source + ')', 'inf');
    }
  }catch(e){
    log('bt toggle error: ' + e, 'err');
  }
}
async function _btLibPin(ev, hash){
  if(ev) ev.stopPropagation();
  try{
    const d = await api('POST','/api/bt/select',{hash});
    if(d && d.pinned){
      _btLibPinned = d.pinned;
      _btLibLastHash = '';
      _btLibRender();
      log('Pinned BT for next execute: ' + (d.instruction || hash), 'ok');
    }else if(d && d.error){
      log('pin failed: ' + d.error, 'err');
    }
  }catch(e){
    log('pin error: ' + e, 'err');
  }
}
async function _btLibUnpin(){
  try{
    await api('POST','/api/bt/unpin',{});
    _btLibPinned = null;
    _btLibLastHash = '';
    _btLibRender();
  }catch(_){}
}
async function _btLibPromote(ev, hash, promoted){
  if(ev) ev.stopPropagation();
  try{
    const d = await api('POST','/api/bt/promote',{hash, promoted});
    if(d && typeof d.promoted === 'boolean'){
      const e = _btLibEntries.find(x => x.hash === hash);
      if(e) e.promoted = d.promoted;
      _btLibLastHash = '';
      _btLibRender();
    }
  }catch(e){
    log('promote error: ' + e, 'err');
  }
}
async function _btLibDelete(ev, hash){
  if(ev) ev.stopPropagation();
  if(!confirm('Delete this cached BT permanently?')) return;
  try{
    const r = await fetch(API + '/api/bt/' + encodeURIComponent(hash), {method:'DELETE'});
    const d = await r.json();
    if(d && d.deleted){
      _btLibEntries = _btLibEntries.filter(x => x.hash !== hash);
      _btLibExpanded.delete(hash);
      if(_btLibPinned === hash) _btLibPinned = null;
      _btLibLastHash = '';
      _btLibRender();
      log('Deleted BT ' + hash, 'inf');
    }else if(d && d.error){
      log('delete failed: ' + d.error, 'err');
    }
  }catch(e){
    log('delete error: ' + e, 'err');
  }
}
// Boot the panel: initial fetch + 10s poll.
setTimeout(_btLibRefresh, 800);
setInterval(_btLibRefresh, 10000);

function _renderDetRows(){
  const box=document.getElementById('det-rows');
  if(!box||!_approvalData)return;
  const dets=(_approvalData.data&&_approvalData.data.detections)||[];
  box.innerHTML=dets.map((det,i)=>{
    const pos=det.position_3d
      ?`(${det.position_3d.map(v=>Number(v).toFixed(3)).join(', ')})`:'no 3D';
    const conf=Math.round((det.confidence||0)*100);
    // Colour the confidence so a weak detection reads as weak at a glance.
    const col=conf>=60?'#6c6':(conf>=30?'#ca6':'#c66');
    return `<div style="display:flex;align-items:center;gap:8px;background:#1a1a1a;border:1px solid #333;border-radius:4px;padding:5px 8px;font-family:monospace;font-size:12px">
      <span style="color:#fff;flex:0 0 190px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${det.label}</span>
      <span style="color:${col};flex:0 0 42px">${conf}%</span>
      <span style="color:#888;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${pos}</span>
      <button onclick="_dropDet(${i})" title="Discard this detection" style="flex:0 0 auto;background:#2a1414;color:#e66;border:1px solid #633;border-radius:4px;cursor:pointer;padding:3px 8px;font-size:13px">&#128465;</button>
    </div>`;
  }).join('')||'<div style="color:#666;font-size:12px">no detections</div>';
}

// Drop one detection from the server's pending set, then re-read what is
// actually left rather than assuming the local edit matched.
async function _dropDet(i){
  const dets=(_approvalData&&_approvalData.data&&_approvalData.data.detections)||[];
  const label=dets[i]?dets[i].label:'?';
  try{
    await api('POST','/api/detections/drop',{indices:[i]});
    const cur=await api('GET','/api/detections',null);
    _approvalData.data.detections=(cur&&cur.detections)||[];
    log('Discarded detection: '+label,'inf');
    _renderDetRows();
  }catch(e){log('Could not discard '+label+': '+e.message,'err')}
}

async function _approveAndExecute(dryOverride){
  // Read the checkbox BEFORE removing the modal it lives in.
  const m=document.getElementById('approval-modal');
  if(m)m.remove();
  const ad=_approvalData;
  if(!ad){log('No approval data','err');return}
  const dry=dryOverride===true?true:(ad.dry||false);
  log((dry?'[DRY] ':'')+'Approved: executing...','inf');
  toast(dry?'Dry run...':'Executing...','inf');
  _showingDetection=false;
  startLiveFeed();
  document.getElementById('cam-tag').textContent=dry?'DRY RUN...':'EXECUTING...';

  // Progress cursor. `since` is a server-side monotonic sequence number, NOT
  // an index into the returned array: the buffer trims from the front on long
  // runs, and the old index-based cursor silently skipped or repeated lines
  // once it did. Seed it from the DELETE so we never replay pre-run entries.
  let _progSince=0;
  try{
    const c=await(await fetch('/api/progress',{method:'DELETE'})).json();
    _progSince=c.next_since||0;
  }catch(e){}
  // _progBusy: skip a tick if the previous poll is still in flight, so slow
  // responses cannot stack up setInterval callbacks against the connection pool.
  let _progBusy=false;
  let progInterval=setInterval(async()=>{
    if(_progBusy) return;
    _progBusy=true;
    try{
      const pg=await(await fetch('/api/progress?since='+_progSince)).json();
      // Render with the server's event time, not the browser's paint time.
      (pg.log||[]).forEach(en=>log(en.msg,'inf',en.ts));
      if(pg.next_since!=null) _progSince=pg.next_since;
    }catch(e){}
    finally{_progBusy=false}
  },1000);
  try{
    const d=await api('POST','/api/execute_approved',
      {instruction:ad.instruction,dry_run:dry});
    clearInterval(progInterval);
    // Plan provenance. During a collection session "llm" means the BT cache
    // missed and Gemini was consulted, so this run is not repeatable.
    if(d.plan_source){
      log('Plan source: '+d.plan_source+(d.bt_hash?' ('+d.bt_hash+')':''),
          d.plan_source==='llm'?'inf':'ok');
    }
    if(d.demo_episode){
      const de=d.demo_episode, cf=(de.spark||{}).commanded_action_fraction;
      // A low commanded fraction means the executor spent the run on the
      // movej/movel path, where no per-timestep command exists and the
      // action array is a post-hoc reconstruction. Flag it, don't hide it.
      log('Demo episode: '+(de.episode_dir||'?')+' ('+(de.num_frames||0)+' frames @ '+
          (de.actual_fps||0)+' Hz'+
          (cf!=null?', '+Math.round(cf*100)+'% commanded action':'')+')',
          (cf!=null&&cf<0.9)?'err':'ok');
    }
    if(d.plan_yaml||d.plan){
      document.getElementById('sec-plan').style.display='block';
      document.getElementById('plan-out').innerHTML=hlYaml(
        d.plan_yaml||(typeof d.plan==='string'?d.plan:JSON.stringify(d.plan,null,2)));
    }
    if(d.detections) showDets(d.detections);
    (d.execution_results||[]).forEach(r=>{
      log('  '+r.action+': '+(r.success?'OK':'FAIL')+' - '+r.message,r.success?'ok':'err');
    });
    log('Task '+(d.success?'completed':'failed')+' in '+(d.duration||0).toFixed(1)+'s',d.success?'ok':'err');
    toast(d.success?'Task done':'Task failed',d.success?'ok':'err',8000);
    addHist(d);
    // Episode bundle panel: only shows after a real (non-dry) run
    // when the recorder produced a cache. Save / Discard buttons
    // are the only way to retain or trash the bundle, starting the
    // next task auto-discards.
    if(d.episode){showEpisodePanel(d.instruction,d.episode,d.success);}
  }catch(e){
    clearInterval(progInterval);
    log('Execution failed: '+e.message,'err');toast('Failed','err',8000);
  }
}
