const API = '';
let tags = [];
let promptMode = 'text';
let drawing = false;
let drawStart = null;

// Tabs
function openTab(t){
  document.querySelectorAll('.tab').forEach(e=>e.classList.remove('on'));
  document.querySelectorAll('.pane').forEach(e=>e.classList.remove('on'));
  document.querySelector(`[data-t="${t}"]`).classList.add('on');
  document.getElementById(`p-${t}`).classList.add('on');
}

// Prompt mode
function setMode(m){
  promptMode=m;
  document.querySelectorAll('.mode-btn').forEach(b=>b.classList.toggle('active',b.dataset.mode===m));
  const hint=document.getElementById('click-hint');
  const sec=document.getElementById('sec-prompts');
  if(m==='click'){
    hint.textContent='Click on an object to detect it';
    hint.style.display='block';
    sec.style.display='none';
  } else if(m==='box'){
    hint.textContent='Draw a box around an object';
    hint.style.display='block';
    sec.style.display='none';
  } else {
    hint.style.display='none';
    sec.style.display='block';
  }
}

// Canvas interaction
function setupCanvas(){
  const vp=document.getElementById('viewport');
  const canvas=document.getElementById('overlay-canvas');
  const img=document.getElementById('cam-img');

  vp.addEventListener('mousedown',e=>{
    if(promptMode==='text') return;
    const rect=img.getBoundingClientRect();
    const x=e.clientX-rect.left, y=e.clientY-rect.top;
    if(x<0||y<0||x>rect.width||y>rect.height) return;
    if(promptMode==='click'){
      doClickDetect(x,y,rect.width,rect.height);
    } else if(promptMode==='box'){
      drawing=true;
      drawStart={x,y,rw:rect.width,rh:rect.height,rx:rect.left,ry:rect.top};
      canvas.width=rect.width; canvas.height=rect.height;
      canvas.style.display='block';
      canvas.style.left=rect.left-vp.getBoundingClientRect().left+'px';
      canvas.style.top=rect.top-vp.getBoundingClientRect().top+'px';
      canvas.style.width=rect.width+'px';
      canvas.style.height=rect.height+'px';
    }
  });

  window.addEventListener('mousemove',e=>{
    if(!drawing||!drawStart) return;
    const canvas=document.getElementById('overlay-canvas');
    const ctx=canvas.getContext('2d');
    const x=e.clientX-drawStart.rx, y=e.clientY-drawStart.ry;
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.strokeStyle='#00d4aa';
    ctx.lineWidth=2;
    ctx.setLineDash([5,4]);
    ctx.strokeRect(drawStart.x,drawStart.y,x-drawStart.x,y-drawStart.y);
    // Corner markers
    ctx.setLineDash([]);
    ctx.fillStyle='#00d4aa';
    [drawStart,{x,y}].forEach(p=>{
      ctx.beginPath();ctx.arc(p.x,p.y,3,0,Math.PI*2);ctx.fill();
    });
  });

  window.addEventListener('mouseup',e=>{
    if(!drawing||!drawStart) return;
    drawing=false;
    const canvas=document.getElementById('overlay-canvas');
    canvas.style.display='none';
    const x=e.clientX-drawStart.rx, y=e.clientY-drawStart.ry;
    if(Math.abs(x-drawStart.x)<10&&Math.abs(y-drawStart.y)<10) return;
    const bx1=Math.min(drawStart.x,x), by1=Math.min(drawStart.y,y);
    const bx2=Math.max(drawStart.x,x), by2=Math.max(drawStart.y,y);
    doBoxDetect(bx1,by1,bx2,by2,drawStart.rw,drawStart.rh);
    drawStart=null;
  });
}

// API
async function api(method,path,body){
  const opts={method,headers:{'Content-Type':'application/json'}};
  if(body) opts.body=JSON.stringify(body);
  const r=await fetch(API+path,opts);
  return r.json();
}

// Status
function setDot(id,on){document.getElementById(id).className='dot '+(on?'on':'off')}

// Robot-family theming
// Display-name table mirrors server-side `_ROBOT_DISPLAY` in routes/core.py
// so /api/status (which only ships the raw family string) can drive a
// theme + chip-label swap without a second round-trip to /api/robot.
const ROBOT_FAMILIES=['ur10e'];
const ROBOT_DISPLAY_NAMES={ur10e:'UR10e',franka:'Franka FR3',bimanual_franka:'Bimanual Franka'};
let _currentRobotFamily=null;
function applyRobotFamily(fam){
  if(!fam) return;
  fam=String(fam).toLowerCase();
  if(!ROBOT_FAMILIES.includes(fam)) return;        // unknown, leave UI alone
  if(fam===_currentRobotFamily) return;            // no-op fast path
  _currentRobotFamily=fam;
  document.body.classList.remove('robot-ur10e');
  document.body.classList.add('robot-'+fam);
  const lbl=document.getElementById('d-robot-label');
  if(lbl) lbl.textContent=ROBOT_DISPLAY_NAMES[fam]||fam;
  // Per-family teleop profile: keyboard map, gamepad button indices,
  // and linear/angular gains. See TELEOP_PROFILES above to retune one
  // family without touching the others. Safe to call before TP exists
  // (early-load order); _setActiveTeleop is a no-op for unknown families.
  if(typeof _setActiveTeleop==='function') _setActiveTeleop(fam);
}

async function pollStatus(){
  try{
    const s=await api('GET','/api/status');
    setDot('d-kinect',s.kinect_connected);
    setDot('d-kinect2',s.kinect2_connected);
    setDot('d-rs',s.realsense_connected);
    setDot('d-robot',s.robot_connected);
    setDot('d-sam3',s.perception_ready);
    // Theme follows backend's reported robot family, covers the case
    // where /api/robot was unavailable on load OR the operator restarts
    // the pipeline with a different --robot flag.
    if(s.robot_family) applyRobotFamily(s.robot_family);
    // Seed the IP input from the server-configured robot_ip (from
    // --ip / configs/<family>_default.yaml) the first time we see one.
    // After the user manually edits the field, leave it alone, they may
    // be testing a different address.
    if(s.robot_ip){
      const ipBox=document.getElementById('in-ip');
      if(ipBox && !ipBox.value && !ipBox.dataset.userTouched){
        ipBox.value=s.robot_ip;
      }
    }
    if(s.initialized){
      const b=document.getElementById('btn-init');
      b.textContent='Ready';b.disabled=true;b.style.opacity='.5';
    }
    updateActivity(s.activity||'idle');
  }catch(e){}
}

function updateActivity(act){
  const pill=document.getElementById('activity-pill');
  const txt=document.getElementById('activity-text');
  if(!pill||!txt) return;
  const labels={idle:'Idle',capturing:'Capturing',detecting:'Detecting',planning:'Planning',executing:'Executing'};
  txt.textContent=labels[act]||act;
  let cls='idle';
  if(act==='executing') cls='executing';
  else if(act==='capturing'||act==='detecting'||act==='planning') cls='thinking';
  pill.className='chip activity-pill '+cls;
}

// Init
async function doInit(){
  const b=document.getElementById('btn-init');
  b.innerHTML='<span class="spinner"></span>';b.disabled=true;
  log('Initializing pipeline...','inf');
  try{
    const r=await api('POST','/api/initialize');
    log('Pipeline initialized','ok');
    toast('Pipeline ready','ok');
    pollStatus();
    doCapture();
  }catch(e){log('Init failed: '+e.message,'err');toast('Init failed','err');b.textContent='Init';b.disabled=false}
}

// Connect robot
async function doConnect(){
  const ip=document.getElementById('in-ip').value;
  const btn=document.getElementById('btn-conn');
  const prog=document.getElementById('conn-prog');
  const fill=document.getElementById('conn-fill');
  const msg=document.getElementById('conn-msg');
  btn.disabled=true;btn.innerHTML='<span class="spinner"></span>';
  prog.style.display='block';
  fill.style.width='0';
  // Use the human-readable robot name from /api/robot (set by --robot
  // family + per-family YAML) so the log line isn't hardcoded UR10e.
  const robotName = (ROBOT_DISPLAY_NAMES && _currentRobotFamily)
    ? ROBOT_DISPLAY_NAMES[_currentRobotFamily] : 'robot';
  msg.textContent='Connecting to '+(ip||'(default)')+'...';
  // Animate progress
  let w=0;
  const iv=setInterval(()=>{w=Math.min(w+Math.random()*15,90);fill.style.width=w+'%'},200);
  log('Connecting to '+robotName+' at '+(ip||'(server-default)')+'...','inf');
  try{
    // Empty ip -> backend uses the server-side configured IP.
    const r=await api('POST','/api/connect_robot',{robot_ip:ip||''});
    clearInterval(iv);
    if(r.connected){
      fill.style.width='100%';msg.textContent='Connected';fill.style.background='var(--green)';
      log(robotName+' connected at '+r.robot_ip,'ok');toast('Robot connected','ok');
      // Reflect the IP the backend actually used back into the input.
      if(r.robot_ip){document.getElementById('in-ip').value=r.robot_ip;}
    } else {
      fill.style.width='100%';fill.style.background='var(--red)';
      msg.textContent=r.error||'Connection failed';
      log('Connection failed: '+(r.error||'unknown'),'err');toast('Connection failed','err');
    }
    pollStatus();
    setTimeout(()=>{prog.style.display='none';fill.style.background='var(--cyan)'},3000);
  }catch(e){
    clearInterval(iv);fill.style.width='100%';fill.style.background='var(--red)';
    msg.textContent=e.message;log('Connection error: '+e.message,'err');
  }
  btn.disabled=false;btn.textContent='Connect';
}
