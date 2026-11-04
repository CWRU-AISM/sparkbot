// Boot
setupCanvas();
setupTeleop();
_setupZoom();
// Poll cadence: 1.5s gives the activity pill (Idle/Detecting/Planning/Executing)
// a responsive feel without hammering the server.
setInterval(pollStatus,1500);
setInterval(refreshTcp,1000);  // 1 Hz; bumping past this hammers the server
pollStatus();
// Fetch robot identity once on load and configure body class + banner.
// Server CLI flag --robot drives this; if /api/robot
// is unavailable (older server build) we silently keep the default UR10e
// banner-less layout. The page is also openable with a ?robot=<family>
// query string for offline / pre-pipeline preview, which seeds the theme
// before either /api/robot or /api/status returns.
function _robotFromQuery(){
  try{
    const q=new URLSearchParams(window.location.search).get('robot');
    if(!q) return null;
    const fam=q.toLowerCase();
    return ROBOT_FAMILIES.includes(fam)?fam:null;
  }catch(e){return null}
}
const _qsFam=_robotFromQuery();
if(_qsFam) applyRobotFamily(_qsFam);
async function applyRobotIdentity(){
  try{
    const r=await fetch('/api/robot');
    if(!r.ok) return;
    const info=await r.json();
    const fam=(info.family||'ur10e').toLowerCase();
    // Body class + chip label. The robot banner is intentionally not shown;
    // the robot chip in the header already identifies the robot.
    applyRobotFamily(fam);
  }catch(e){/* leave default UI in place; pollStatus will retry via robot_family */}
}
applyRobotIdentity();
// Auto-start live feed after 2s (give pipeline time to respond)
setTimeout(startLiveFeed,2000);
