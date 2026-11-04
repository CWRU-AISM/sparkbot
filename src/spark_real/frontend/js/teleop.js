// Teleop: keyboard + gamepad (per-family profiles)
// Each robot family has its own profile so the keyboard map, gamepad
// button layout, linear/angular gains, and per-axis sign flips can be
// tuned independently. The active profile is selected by
// applyRobotFamily() and lives in TP. To tweak bindings for one robot
// without touching the others, edit only that family's entry below.
function _defaultKeys(linVel, angVel, downVel){
  // Standard IJKL/UO linear + WASD/QE rotational layout. Re-use this
  // in a profile via `keys: _defaultKeys(profile.linVel, profile.angVel)`
  // or replace the field with a hand-rolled map for fully custom bindings.
  // `downVel` (m/s, optional) separately tunes the descent gain on `o`.
  // Lets operators ease the gripper down onto a target without
  // overshoot. Falls back to linVel.
  const dn = (downVel === undefined) ? linVel : downVel;
  return {
    i:[ linVel,0,0,0,0,0],   // fwd  (+X)
    k:[-linVel,0,0,0,0,0],   // back (-X)
    j:[0, linVel,0,0,0,0],   // left (+Y)
    l:[0,-linVel,0,0,0,0],   // right(-Y)
    u:[0,0, linVel,0,0,0],   // up   (+Z)
    o:[0,0,    -dn,0,0,0],   // down (-Z, separately tunable via downVel)
    w:[0,0,0, angVel,0,0],   // pitch up   (+Rx)
    s:[0,0,0,-angVel,0,0],   // pitch down (-Rx)
    a:[0,0,0,0, angVel,0],   // roll left  (+Ry)
    d:[0,0,0,0,-angVel,0],   // roll right (-Ry)
    q:[0,0,0,0,0, angVel],   // yaw left   (+Rz)
    e:[0,0,0,0,0,-angVel],   // yaw right  (-Rz)
  };
}

const TELEOP_PROFILES={
  ur10e:{
    linVel: 0.10,            // m/s   commanded linear velocity per fully-deflected stick or key
    angVel: 0.50,            // rad/s commanded angular velocity per fully-deflected stick or key
    deadzone: 0.15,          // gamepad stick deadzone (0..1)
    axisSigns: [1,1,1,1,1,1],// per-component sign multiplier on the final 6-vector [vx,vy,vz,wrx,wry,wrz]
    get keys(){ return _defaultKeys(this.linVel, this.angVel); },
    gamepad:{
      leftStickX: 0,         // axis index for left stick X (drives vel[1], "left/right")
      leftStickY: 1,         // axis index for left stick Y (drives vel[0], "fwd/back")
      rightStickX: 2,        // axis index for right stick X (drives vel[5], "yaw")
      rightStickY: 3,        // axis index for right stick Y (drives vel[3], "pitch")
      invertLeftY: true,     // most pads report stick-up as negative; true keeps "up = forward"
      invertLeftX: true,     // true keeps "stick-left = +Y"
      invertRightX: true,    // true keeps "stick-left = +yaw"
      invertRightY: true,    // true keeps "stick-up = +pitch"
      dpadUp: 12, dpadDown: 13, // Z+ / Z-
      bumperL: 4, bumperR: 5,   // roll+ / roll-
      triggerClose: 6, triggerOpen: 7, // gripper close / open (rising edge)
      homeBtn: 8,            // home pose
      recoverBtn: 9,         // clear reflex / protective stop without moving
    },
  },
};

// Active profile. Updated by applyRobotFamily(). Default to UR10e so the
// page works during the brief window before /api/status reports a family.
let TP = TELEOP_PROFILES.ur10e;
function _setActiveTeleop(fam){
  const p = TELEOP_PROFILES[fam];
  if(p) TP = p;
}

let teleopActive=false;
let teleopLocked=false;  // sticky-mode flag; set true on click, false on STOP
let heldKeys=new Set();
let teleopInterval=null;
let gripperOpen=true;
let gamepadConnected=false;
let gamepadIdx=null;
let gamepadMapping='';     // 'standard' when the browser gives the canonical
                           // W3C layout the profile indices assume; '' (raw
                           // HID) otherwise - raw ordering varies by USB port,
                           // which is why buttons move (Home -> speed etc).
let gamepadLogInput=false; // when true, log every pressed button / moved axis
                           // index so a non-standard pad can be re-mapped.
let _gpPrevButtons=[];

function setupTeleop(){
  const zone=document.getElementById('teleop-zone');
  zone.addEventListener('keydown',e=>{
    const k=e.key.toLowerCase();
    if(k==='escape'){doStop();return}
    if(k==='h'){doHome();return}
    if(k==='g'){
      gripperOpen=!gripperOpen;
      grip(gripperOpen?'open':'close');
      return;
    }
    if(TP.keys[k]&&!heldKeys.has(k)){
      heldKeys.add(k);
      updateKeyIndicators();
    }
    e.preventDefault();
  });
  zone.addEventListener('keyup',e=>{
    const k=e.key.toLowerCase();
    heldKeys.delete(k);
    updateKeyIndicators();
  });
  zone.addEventListener('focus',()=>{
    // Clicking the zone arms teleop and LOCKS it sticky, survives blur
    // (e.g., clicking "Add Anchor" elsewhere on the page) so the user
    // can drive between anchor points without re-clicking each time.
    // STOP / ESC clears teleopLocked.
    teleopActive=true;
    teleopLocked=true;
    document.getElementById('teleop-status').innerHTML='<span class="live-dot"></span>Teleop LOCKED: STOP/Esc to release';
    /* focus color handled via accent */
    
    if(!teleopInterval){
      teleopInterval=setInterval(teleopTick,33); // 50 Hz, smoother than the old 30 Hz
    }
  });
  zone.addEventListener('blur',()=>{
    if(teleopLocked){
      // Sticky: keep teleop interval + gamepad polling alive. Drop any
      // held keys (since keydown/keyup won't fire without focus) and
      // hint to the user that re-focusing restores keyboard.
      heldKeys.clear();
      updateKeyIndicators();
      sendStopVelocity();
      document.getElementById('teleop-status').innerHTML='<span class="live-dot"></span>Teleop LOCKED (gamepad live, click to regain keys): STOP to release';
      return;
    }
    teleopActive=false;
    heldKeys.clear();
    updateKeyIndicators();
    document.getElementById('teleop-status').textContent='Click to enable teleop';
    
    
    if(teleopInterval){clearInterval(teleopInterval);teleopInterval=null}
    sendStopVelocity();
  });

  // Gamepad events
  window.addEventListener('gamepadconnected',e=>{
    gamepadConnected=true; gamepadIdx=e.gamepad.index;
    gamepadMapping=e.gamepad.mapping||'';
    const name=e.gamepad.id;
    const std=gamepadMapping==='standard';
    // The profile button/axis indices are the W3C "standard" layout. When the
    // browser reports a non-standard mapping the raw HID order applies instead
    // and differs by USB port, so the indices land on the wrong controls.
    const stat=document.getElementById('gamepad-status');
    stat.textContent='Gamepad: '+name+'  ['+(std?'standard \u2713':'NON-STANDARD \u26a0')+
      '  '+e.gamepad.buttons.length+'btn/'+e.gamepad.axes.length+'ax]';
    stat.style.color=std?'var(--tx-3)':'var(--amber)';
    log('Gamepad: '+name+' | mapping='+(gamepadMapping||'(raw)')+
        ' | '+e.gamepad.buttons.length+' buttons, '+e.gamepad.axes.length+' axes', std?'ok':'err');
    if(!std) log('\u26a0 Non-standard mapping: button indices are unreliable and '+
                 'change by USB port. Try another port for "standard", or turn on '+
                 '"log gamepad input" and press each button to re-map.','err');
  });
  window.addEventListener('gamepaddisconnected',()=>{
    gamepadConnected=false; gamepadIdx=null;
    document.getElementById('gamepad-status').textContent='';
    log('Gamepad disconnected','inf');
  });
}

function updateKeyIndicators(){
  document.querySelectorAll('.key-ind').forEach(el=>{
    el.classList.toggle('pressed',heldKeys.has(el.dataset.k));
  });
}

function readGamepad(){
  // Returns [vx,vy,vz,wrx,wry,wrz] from gamepad, or null. All bindings
  // (linear/angular gains, deadzone, stick axes, button indices, sign
  // flips) come from the active per-family profile TP, see TELEOP_PROFILES.
  if(!gamepadConnected||gamepadIdx===null) return null;
  const gp=navigator.getGamepads()[gamepadIdx];
  if(!gp) return null;
  // Re-map helper: with logging on, report the raw index of any button that
  // was just pressed (and any strongly-deflected axis) so a non-standard pad
  // can be mapped to the profile. Press Home, read "button N pressed".
  if(gamepadLogInput){
    gp.buttons.forEach((b,i)=>{
      const was=_gpPrevButtons[i];
      if(b.pressed && !was) log('gamepad button '+i+' pressed','acc');
    });
    gp.axes.forEach((v,i)=>{ if(Math.abs(v)>0.7) log('gamepad axis '+i+' = '+v.toFixed(2),'acc'); });
  }
  _gpPrevButtons=gp.buttons.map(b=>b.pressed);
  const g = TP.gamepad;
  const LIN = TP.linVel, ANG = TP.angVel;
  const vel=[0,0,0,0,0,0];
  const dz=TP.deadzone;
  const ax=i=>{const v=gp.axes[i]||0; return Math.abs(v)<dz?0:v};
  const sx = s => s ? -1 : 1;  // invert flag -> sign multiplier

  // Left stick -> XY linear
  vel[0]=sx(g.invertLeftY)*ax(g.leftStickY)*LIN; // forward/back
  vel[1]=sx(g.invertLeftX)*ax(g.leftStickX)*LIN; // left/right

  // Right stick -> rotation
  vel[5]=sx(g.invertRightX)*ax(g.rightStickX)*ANG; // yaw
  vel[3]=sx(g.invertRightY)*ax(g.rightStickY)*ANG; // pitch

  // D-pad -> Z. Many gamepads (Logitech F710 X-input, some PS controllers
  // in DirectInput mode) emit D-pad as buttons[12..15]; others (PS5
  // DualSense, some Xbox driver versions) emit as axes[6]/axes[7]. Check
  // BOTH so the D-pad works regardless of which mapping the browser
  // exposes.
  let dpadVz = 0;
  const _bUp   = gp.buttons[g.dpadUp];
  const _bDown = gp.buttons[g.dpadDown];
  if(_bUp?.pressed)   dpadVz =  LIN;
  if(_bDown?.pressed) dpadVz = -LIN;
  // Axes fallback. Standard mappings put D-pad Y on axes[7] (Xbox X-input,
  // most modern controllers) or axes[5] (DirectInput Logitech, some older
  // pads). Same convention either way: -1 = up, +1 = down. Check both
  // so the same code works across controllers without per-pad profiles.
  const dpadAxX = gp.axes[6] !== undefined ? gp.axes[6] : 0;
  const dpadAxY7 = gp.axes[7] !== undefined ? gp.axes[7] : 0;
  const dpadAxY5 = gp.axes[5] !== undefined ? gp.axes[5] : 0;
  // Pick the larger-magnitude axis (the one the pad is actually driving).
  const dpadAxY = Math.abs(dpadAxY7) >= Math.abs(dpadAxY5) ? dpadAxY7 : dpadAxY5;
  if(dpadVz === 0 && Math.abs(dpadAxY) > 0.5){
    dpadVz = dpadAxY < 0 ? LIN : -LIN;  // axis up = -1 -> +Z
  }
  if(dpadVz !== 0) vel[2] = dpadVz;

  // Bumpers -> roll
  if(gp.buttons[g.bumperL]?.pressed) vel[4]= ANG;
  if(gp.buttons[g.bumperR]?.pressed) vel[4]=-ANG;

  // Gripper: simple binary edge-detected close/open (mirrors the
  // UR10e cell mapping which has been reliable for months).
  // Button indices come from the per-family profile (g.triggerClose /
  // g.triggerOpen, usually 6 and 7 on standard Xbox layouts).
  // Rising-edge detection only, no analog streaming, no internal
  // state to drift out of sync.  Each press fires one /api/gripper
  // close or open with the default driver force.
  const _closePressed = !!gp.buttons[g.triggerClose]?.pressed;
  const _openPressed  = !!gp.buttons[g.triggerOpen]?.pressed;
  if (_closePressed && !readGamepad._closePrev) grip('close');
  if (_openPressed  && !readGamepad._openPrev)  grip('open');
  readGamepad._closePrev = _closePressed;
  readGamepad._openPrev  = _openPressed;

  // Home button (rising edge so a 30 Hz teleop tick doesn't spam /api/home
  // for as long as the button is held).
  const homeNow = !!gp.buttons[g.homeBtn]?.pressed;
  if(homeNow && !readGamepad._homePrev) doHome();
  readGamepad._homePrev = homeNow;

  // Recover button (rising edge). Distinct from home: clears reflex /
  // protective-stop without moving the arm. Use after singularity or
  // joint-limit lockup so teleop resumes from the current pose.
  if(g.recoverBtn!==undefined){
    const recNow = !!gp.buttons[g.recoverBtn]?.pressed;
    if(recNow && !readGamepad._recoverPrev) doRecover();
    readGamepad._recoverPrev = recNow;
  }

  // Per-family final sign flip on the 6-vector (e.g. flip "forward" axis).
  const s = TP.axisSigns || [1,1,1,1,1,1];
  for(let i=0;i<6;i++) vel[i]*=s[i];

  return vel;
}

function teleopTick(){
  // Combine keyboard + gamepad into one velocity
  const vel=[0,0,0,0,0,0];

  // Keyboard (per-family profile keys, with final axis-sign flip applied)
  const keyMap = TP.keys;
  const signs = TP.axisSigns || [1,1,1,1,1,1];
  for(const k of heldKeys){
    const v=keyMap[k];
    if(v) for(let i=0;i<6;i++) vel[i]+=v[i]*signs[i];
  }

  // Gamepad (readGamepad already applies the profile and axis signs)
  const gpVel=readGamepad();
  if(gpVel) for(let i=0;i<6;i++) vel[i]+=gpVel[i];

  // Velocity dispatch policy:
  //   POST every tick WHILE moving (so EMA tracks input + franky's
  //   CartesianVelocityMotion keeps re-planning to the live target).
  //   POST ONCE on the moving->stopped transition (sends an explicit
  //   zero so the EMA decays + franky halts).
  //   Do NOT post when idle. Earlier we tried "always POST while
  //   teleopActive" to fix a "won't stop" bug, but that hammered
  //   franky with thousands of zero-velocity commands per session,
  //   eventually corrupting its internal state and triggering a
  //   C++ `terminate called without an active exception` crash that
  //   takes the whole server down.
  const moving = vel.some(v=>Math.abs(v) > 1e-4);
  if(moving){
    api('POST','/api/velocity',{vx:vel[0],vy:vel[1],vz:vel[2],wrx:vel[3],wry:vel[4],wrz:vel[5],duration:0.05}).catch(()=>{});
    window._teleopWasMoving = true;
  } else if(window._teleopWasMoving){
    // Falling-edge: send one explicit zero so franky receives a
    // halt command (the duration=0.05 of the previous motion would
    // have stopped naturally too, but this is belt-and-suspenders).
    api('POST','/api/velocity',{vx:0,vy:0,vz:0,wrx:0,wry:0,wrz:0,duration:0.05}).catch(()=>{});
    window._teleopWasMoving = false;
  }
}

async function sendStopVelocity(){
  try{await api('POST','/api/velocity',{vx:0,vy:0,vz:0,wrx:0,wry:0,wrz:0,duration:0.01})}catch(e){}
}

// Click-based jog (for remote desktop / touch)
let _jogIntervals={};
function jogStart(k){
  if(_jogIntervals[k]) return;
  if(!teleopActive){
    teleopActive=true;
    document.getElementById('teleop-status').innerHTML='<span class="live-dot"></span>Teleop active';
    if(!teleopInterval) teleopInterval=setInterval(teleopTick,33);
  }
  heldKeys.add(k);
  updateKeyIndicators();
  // Also send immediately
  teleopTick();
}
function jogStop(k){
  heldKeys.delete(k);
  updateKeyIndicators();
  if(heldKeys.size===0) sendStopVelocity();
}
