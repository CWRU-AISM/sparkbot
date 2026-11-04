// Robot controls
async function jog(dx,dy,dz){
  try{await api('POST','/api/move_relative',{dx,dy,dz});refreshTcp()}
  catch(e){log('Jog: '+e.message,'err')}
}
async function doHome(){log('Homing...','inf');if(teleopInterval){clearInterval(teleopInterval);teleopInterval=null}heldKeys.clear();try{await api('POST','/api/home');log('Home','ok');refreshTcp()}catch(e){log('Home: '+e.message,'err')}if(teleopActive&&!teleopInterval){teleopInterval=setInterval(teleopTick,33)}}

async function doRecover(){
  // Clear reflex / protective-stop without moving the arm. Distinct from
  // doHome (which physically moves to ready pose) and doStop (which only
  // brakes). Pauses the teleop tick so a held stick doesn't immediately
  // re-trip the reflex; resumes after the call returns.
  const wasActive=!!teleopInterval;
  if(teleopInterval){clearInterval(teleopInterval);teleopInterval=null}
  try{
    const r=await api('POST','/api/recover');
    log('Recovered: '+(r.actions||[]).join(', '),'ok');
  }catch(e){log('Recover: '+e.message,'err')}
  if(teleopActive&&wasActive&&!teleopInterval){
    teleopInterval=setInterval(teleopTick,33);
  }
}
async function grip(a){try{await api('POST','/api/gripper',{action:a});log('Gripper '+a,'ok')}catch(e){log('Gripper: '+e.message,'err')}}

// Streaming variable-width gripper command for analog triggers.
// Throttle: 100 ms between POSTs AND 0.02 change in target, chosen so
// the gripper actually completes one
// motion before the next supersedes it.  `force=true` bypasses the
// throttle entirely; used on trigger release to commit a tap's final
// accumulated target even if shorter than the throttle window.
let _gripLastSent=1.0;       // 1.0 = closed at rest; opens as left trigger pulls
let _gripLastSentAt=0;
function sendGripperPosition(target, force){
  if(target == null) return;
  const now=performance.now();
  if(!force){
    if(now - _gripLastSentAt < 100) return;
    if(Math.abs(target - _gripLastSent) < 0.02) return;
  }
  _gripLastSent = target;
  _gripLastSentAt = now;
  gripperOpen = (target < 0.5);
  api('POST','/api/gripper_position',{position:target}).catch(()=>{});
}
async function testYaw(){log('Running yaw test (0/30/60 deg)...','inf');try{const r=await api('POST','/api/test_yaw');if(r.error){log('Yaw test: '+r.error,'err');return}if(r.results){r.results.forEach(d=>log(`Yaw ${d.yaw_deg} deg: tool_X=${d.tool_X_deg}, tool_Y=${d.tool_Y_deg}`,'ok'))}log('Yaw test done','ok')}catch(e){log('Yaw test: '+e.message,'err')}}
async function logPose(){try{const r=await api('GET','/api/robot_state');if(r.tcp_pose){const p=r.tcp_pose;log(`TCP: [${p[0].toFixed(3)}, ${p[1].toFixed(3)}, ${p[2].toFixed(3)}] ori=[${p[3].toFixed(3)}, ${p[4].toFixed(3)}, ${p[5].toFixed(3)}]`,'inf');const j=r.joint_positions;if(j){const jd=j.map(v=>(v*180/Math.PI).toFixed(1));log(`Joints (deg): [${jd.join(', ')}]`,'inf')}log(`Gripper pos: ${r.gripper_position}`,'inf')}else{log('No TCP data','err')}}catch(e){log('Pose: '+e.message,'err')}}

async function doStop(opts){
  const silent=!!(opts&&opts.silent);
  if(!silent) log('EMERGENCY STOP','err');
  // Belt-and-suspenders: send an immediate zero-velocity BEFORE /api/stop.
  // Reason: /api/stop calls the driver's stop() which halts the active
  // motion, but any in-flight CartesianVelocityMotion with duration>0
  // may still finish out its remaining duration. An explicit zero
  // velocity tells franky "command zero RIGHT NOW" so even the
  // remaining duration is a no-op. Fire-and-forget so we don't block
  // the rest of the stop cleanup.
  try{
    api('POST','/api/velocity',{vx:0,vy:0,vz:0,wrx:0,wry:0,wrz:0,duration:0.05}).catch(()=>{});
  }catch(e){}
  // Release the sticky-teleop lock and tear down the tick loop so the
  // STOP/Esc buttons inside the teleop block are the canonical "leave
  // teleop" path.  Side-effect of any /api/stop call, consistent with
  // every other reason a user would hit this.
  teleopLocked=false;
  teleopActive=false;
  heldKeys.clear();
  updateKeyIndicators();
  if(teleopInterval){clearInterval(teleopInterval);teleopInterval=null}
  const zone=document.getElementById('teleop-zone');
  if(zone){
    zone.style.borderColor='var(--border-hi)';
    zone.style.borderStyle='dashed';
    zone.blur();
  }
  const st=document.getElementById('teleop-status');
  if(st) st.textContent='Click to enable teleop';
  try{
    await api('POST','/api/stop');
    if(!silent) toast('Stopped - use FAIL on the episode bar to keep it as a negative','err',5000);
  }catch(e){}
}
// Convert an axis-angle rotvec (x,y,z) to a 3x3 rotation matrix.
// Rodrigues' formula. Returns a 9-element row-major array for cheap access.
function _rotvecToMat(rx, ry, rz){
  const ang = Math.sqrt(rx*rx + ry*ry + rz*rz);
  if(ang < 1e-9) return [1,0,0, 0,1,0, 0,0,1];
  const x = rx/ang, y = ry/ang, z = rz/ang;
  const c = Math.cos(ang), s = Math.sin(ang), C = 1-c;
  return [
    c + x*x*C,    x*y*C - z*s, x*z*C + y*s,
    y*x*C + z*s, c + y*y*C,    y*z*C - x*s,
    z*x*C - y*s, z*y*C + x*s, c + z*z*C,
  ];
}

async function refreshTcp(){
  try{
    const s=await api('GET','/api/robot_state');
    if(s.connected&&s.tcp_pose){
      document.getElementById('tcp-x').textContent=s.tcp_pose[0].toFixed(4);
      document.getElementById('tcp-y').textContent=s.tcp_pose[1].toFixed(4);
      document.getElementById('tcp-z').textContent=s.tcp_pose[2].toFixed(4);
      // World-frame TCP axis angles + tilt-from-vertical for orientation
      // self-monitoring during teleop alignment. Decompose the rotvec
      // and read TCP-X, TCP-Y as their XY-projected world angles.
      // Tilt is the angle of TCP-Z from world -Z
      // (i.e., how much the gripper is off straight-down).
      const rv = s.tcp_pose.slice(3);
      const M = _rotvecToMat(rv[0], rv[1], rv[2]);
      // TCP-X world: column 0
      const txx = M[0], txy = M[3], txz = M[6];
      // TCP-Y world: column 1
      const tyx = M[1], tyy = M[4], tyz = M[7];
      // TCP-Z world: column 2
      const tzx = M[2], tzy = M[5], tzz = M[8];
      const rad = Math.PI/180;
      const txAng = Math.atan2(txy, txx) / rad;
      const tyAng = Math.atan2(tyy, tyx) / rad;
      // Mod 180 (axis-undirected) for the side display
      const wrap180 = a => { let v = a % 180; if(v < -90) v += 180; if(v > 90) v -= 180; return v; };
      const tilt = Math.acos(Math.max(-1, Math.min(1, -tzz))) / rad;  // -tzz because we want tilt from -Z
      const fmt = (v, w) => `${v.toFixed(1)} deg (mod180: ${w.toFixed(1)} deg)`;
      document.getElementById('tcp-rx').textContent = fmt(txAng, wrap180(txAng));
      document.getElementById('tcp-ry').textContent = fmt(tyAng, wrap180(tyAng));
      document.getElementById('tcp-tilt').textContent = `${tilt.toFixed(1)} deg`;
      // Gripper position uses the normalized 0..1 scale from /api/robot_state
      // (gripper_position_norm; 0=open, 1=closed). Backstop: if only the raw
      // Robotiq 0..255 value is present, treat <128 as Open.
      (function(){
        const gN = s.gripper_position_norm;
        const gR = s.gripper_position;
        let isOpen = false;
        if (typeof gN === 'number')      isOpen = gN < 0.5;
        else if (typeof gR === 'number') isOpen = gR < 128;
        document.getElementById('tcp-g').textContent = isOpen ? 'Open' : 'Closed';
      })();
    }
  }catch(e){}
}
