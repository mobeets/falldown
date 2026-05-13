let levels = [];
let levelIndex = 0;
let cameraY = 0;
let cameraYTarget = 0;
let cameraMode; // options: 0 = 'follow', 1 = 'drift'

let gravity;
let ballAccel;      // acceleration added by pressing key
let modeSwitchCooldown; // min levels per cameraMode
let levelWidth;     // total width in pixels
let levelHeight;   // vertical height of each level (i.e. vertical distance between platforms)
let levelStartX;    // x value where levels start (to center them)
let levelEndX;      // x value where levels end
let levelSpacing;   // vertical distance between levels
let scrollSpeed;     // speed at which camera drifts up in scroll mode
let initScrollSpeed; // initial scroll speed, used for calculating acceleration of scroll speed if applicable
let maxScrollSpeed;  // max scroll speed, used if scrollSpeed increases over time

let ball;
let experiment;
let trial_block;
let trial;
let config;
let photodiode;
let controls;
let user;
let clickSound;
let E;
let myFont;
let lastScore;

const PLAY_MODE = 0;
const PAUSE_MODE = 1;
const STARTING_MODE = 2;
const READY_MODE = 3;
const COMPLETE_MODE = 4;
let gameMode = READY_MODE;

// todo: track pause times

function preload() {
  clickSound = new Audio('static/click.mp3');
  myFont = loadFont('static/LuckiestGuy-Regular.ttf');
  config = loadConfig();
}

function setup() {
  let cnv = createCanvas(windowWidth, windowHeight);
  cnv.parent('canvas-container'); // attach to the centered div
  textFont(myFont);
  E = new Experiment(config);
  
  levelWidth = E.params.levelWidthProportion * windowWidth;
  levelHeight = E.params.levelHeight * (levelWidth / 600);
  levelStartX = (windowWidth - levelWidth) / 2;
  levelEndX = levelStartX + levelWidth;
  
  cameraMode = E.params.startCameraMode;

  photodiode = new Photodiode(E.params.photodiode, width, height);
  controls = new UnifiedControls(wsLogger);
  user = new TaskControls(controls);

  // adjust gravity and ballAccel relative to 600x600 window
  gravity = E.params.relativeGravity * (levelWidth / 600);
  ballAccel = E.params.relativeBallAccel * (levelWidth / 600);
  initScrollSpeed = E.params.scrollSpeed * (height / 600);
  maxScrollSpeed = E.params.maxScrollSpeed * (height / 600);
  deltaScrollSpeed = (maxScrollSpeed - initScrollSpeed) / (E.params.scrollSpeedSecsToMax * E.params.FPS);

  // set level spacing so that the same number of levels are visible
  levelSpacing = height / E.params.nLevelsVisible;

  let gapSize = levelWidth / E.params.nSegments;
  ball = new Ball(levelWidth/2, 100, 0.1*gapSize);
  E.gameInfo = getGameInfo();
  newGame(false);
}

function newGame(restartGame = false, goBack = false) {
  if (trial_block !== undefined) {
    lastScore = `${trial_block.trials.length} of ${trial_block.block_config.levels.length}`;
  }
  trial_block = E.next_block(restartGame, goBack);
  console.log("Current block config:", trial_block);
  if (trial_block === undefined) { gameMode = COMPLETE_MODE; return; }
  gameMode = READY_MODE;
  
  // Set ball position
  ball.x = levelWidth/2;
  ball.y = 100;
  ball.vx = 0;
  ball.vy = 0;
  cameraY = 0;
  cameraYTarget = 0;
  modeSwitchCooldown = E.params.minLevelsPerMode;
  cameraMode = E.params.startCameraMode;
  scrollSpeed = initScrollSpeed;
  
  // Create initial levels
  levels = [];
  levelIndex = 0;
  let prevTrial;
  for (let i = 0; i < 10; i++) {
    trial = trial_block.next_trial();
    if (trial === undefined) continue;

    let y = height/2 + i * levelSpacing;
    levelIndex++;

    levels.push(new Level(levelIndex, E.params.nSegments, levelWidth, levelHeight, trial, levelStartX, y, cameraMode, E.params.modeRectColors[cameraMode], false));
    prevTrial = trial;
  }
}

function checkForModeSwitch(modeIndex, modeSwitchRate) {
  let doModeSwitch = false;
  if (modeSwitchCooldown > 0) {
    modeSwitchCooldown--;
  } else {
    doModeSwitch = random() < modeSwitchRate;
    if (doModeSwitch) modeSwitchCooldown = E.params.minLevelsPerMode;
  }
  if (doModeSwitch) modeIndex = int(!modeIndex);
  return {modeIndex, doModeSwitch};
}

function decisionEvent(level) {
  let event = level.toJSON();
  event.cameraMode = cameraMode;
  event.ballX = ball.x;
  event.ballY = ball.y;
  event.cameraY = cameraY;
  event.scrollSpeed = scrollSpeed;
  return event;
}

function draw() {
  frameRate(E.params.FPS);
  background(40);
  fill(0); noStroke(); rect(0, 0, levelStartX, height);
  fill(0); noStroke(); rect(levelEndX, 0, windowWidth - levelEndX, height);

  controls.update();
  checkUserButtonPresses();

  if (gameMode == PLAY_MODE) {

    if (user.moveLeft) {
      if (E.params.isMomentum === true || E.params.isMomentum === undefined) {
        ball.vx -= ballAccel;
      } else {
        ball.vx = -E.params.maxBallAccelScale*ballAccel;
      }
      trial_block.log_user_input(-1);
    } else if (user.moveRight) {
      if (E.params.isMomentum === true || E.params.isMomentum === undefined) {
        ball.vx += ballAccel;
      } else {
        ball.vx = E.params.maxBallAccelScale*ballAccel;
      }
      trial_block.log_user_input(1);
    } else {
      if (!E.params.isMomentum) {
        ball.vx = 0;
      }
    }

    ball.vx = constrain(ball.vx, -E.params.maxBallAccelScale*ballAccel, E.params.maxBallAccelScale*ballAccel);
    trial_block.log_states(ball); // logs ball and camera states
    ball.update();

    // Set y offset based on camera mode
    if (cameraMode === 0) {
      // keep ball halfway up screen, but smooth movements
      cameraY = 0.25*(ball.y - height/2) + 0.75*cameraY;
      cameraYTarget = cameraY;
    } else if (cameraMode === 1) {
      // cameraY += scrollSpeed;      
      // if ball is in lower part of screen, keep camera fixed on ball so that ball can't go lower, but continue to update cameraYTarget so that when ball goes back up it will be back in the right place
      if (ball.y - cameraYTarget > 3*height/5) {
        cameraY = 0.1*(ball.y - 3*height/5) + 0.9*cameraY;
        cameraYTarget = cameraY;
      } else {
        cameraY = cameraYTarget;
      }      
      cameraYTarget += scrollSpeed;
    }
    if (cameraMode >= 1 && scrollSpeed < maxScrollSpeed) {
      scrollSpeed += deltaScrollSpeed;
    }
    scrollSpeed = constrain(scrollSpeed, initScrollSpeed, maxScrollSpeed);

    // Check for any block instructions
    let blockInstrs = trial_block.block_config.params.instructions;
    if (blockInstrs !== undefined && blockInstrs.length > 0) {
      showInstructions(blockInstrs, 50);
    }
    
    // Update and render levels
    for (let lvl of levels) {
      lvl.update();
      lvl.render();
      lvl.collidesWith(ball);
      if (lvl.passedThrough(ball)) {
        // trial_block.add_trial(lvl);
        lvl.trial.trigger(decisionEvent(lvl));
        trial_block.last_trial_completed = lvl.trial.index+1;
        markEvent(); // trigger photodiode and play sound
        // toggle mode when we pass through
        if (lvl.isModeSwitch) cameraMode = lvl.modeIndex; //int(!cameraMode);
      }
    }
  
    // Remove levels that went off top and add new ones at bottom
    if (levels.length > 0 && levels[0].y - cameraY < -50) {
      levels.shift();
      
      // Set params for new level
      levelIndex++;
      let trial = trial_block.next_trial();
      if (trial !== undefined) {
        let newY = levels[levels.length - 1].y + levelSpacing;
        let modeIndex = levels[levels.length - 1].modeIndex;
  
        // Check for mode switch on this level
        let modeInfo = checkForModeSwitch(modeIndex, E.params.modeSwitchRates[modeIndex]);
        
        // Create new level
        levels.push(new Level(levelIndex, E.params.nSegments, levelWidth, levelHeight, trial, levelStartX, newY, modeInfo.modeIndex, E.params.modeRectColors[modeInfo.modeIndex], modeInfo.doModeSwitch));
      }
    }
    if (trial_block.is_complete()) { // all levels have been completed
      newGame(false);
    }

    // Render ball
    ball.render();
    
    // Game over condition
    if ((cameraMode === 1) && (ball.y - cameraY < 0)) {
      newGame(false);
    }
    
    drawHUD();

  } else {
    drawPauseScreen();
  }
  
  // render photodiode last
  photodiode.update();
  photodiode.render();
}

function formatTime(seconds) {
  const m = Math.floor(seconds / 60).toString().padStart(2, '0');
  const s = Math.floor(seconds % 60).toString().padStart(2, '0');
  const tenth = Math.floor((seconds % 1) * 10);
  return `${m}:${s}.${tenth}`;
}

function drawTimer(elapsedTimeMsecs) {
  if (isNaN(elapsedTimeMsecs) || elapsedTimeMsecs === undefined) {
    return;
  }
  fill(255); 
  noStroke();
  textSize(20);
  textAlign(CENTER, CENTER);
  const cx = levelStartX / 2;
  const cy = 20;
  let timerText = formatTime(elapsedTimeMsecs / 1000);
  text(timerText, cx, cy);
}

function drawCompletionWedge(pct) {
  const cx = levelStartX / 2;
  const cy = cx;
  const r = Math.floor(levelStartX / 4);

  const startAngle = -HALF_PI; // 12 o'clock
  const endAngle = -HALF_PI + TWO_PI * (pct / 100);

  // Green wedge
  fill(0, 200, 100);
  noStroke();
  arc(cx, cy, r * 2, r * 2, startAngle, endAngle, PIE);

  // White border (circle outline only)
  noFill();
  stroke(255);
  strokeWeight(3);
  ellipse(cx, cy, r * 2, r * 2);
}

function drawHUD() {
  if (trial_block === undefined) return;
  let elapsedTimeMsecs = trial_block.get_elapsed_time();
  drawTimer(elapsedTimeMsecs);

  let totalTrials = trial_block.block_config.levels.length;
  let currentTrial = trial_block.last_trial_completed;
  let progressPct = 0;
  if (totalTrials > 0) {
      progressPct = constrain(currentTrial / totalTrials, 0, 1);
  }
  drawCompletionWedge(progressPct * 100);
}


function showInstructions(text_lines, yOffset) {
  textSize(20);
  textFont('arial');
  fill('white');
  for (var i = 0; i < text_lines.length; i++) {
    // text wrapping seems to ignore centering
    // so to get a width/2 text box centered in screen
    //  we constrain width to width/2, and start text at width/4
    text(text_lines[i], width / 4, yOffset + 60*i, width / 2);
  }
  textFont(myFont);
}

function drawPauseScreen() {
  textAlign(CENTER, CENTER);
  // fill(color(50, 50, 50, 200));
  // rect(0, 0, width, height);
  fill('white');
  textSize(48);

  let firstLineY = 2 * height / 9;
  let secondLineY = 3 * height / 9;
  let thirdLineY = 4 * height / 9;
  
  let controlInstruction = "Use the left joystick";
  let nextButton = "START";
  if (E.params.isCloudStudy) {
    controlInstruction = "Press the left and right arrow keys";
    nextButton = "spacebar";
  }
  let gameInstrStr = "In this task you will attempt to move a ball through different mazes as quickly as possible.";
  let nextInstrStr = "Press " + nextButton + " to continue.";
  let welcomeInstrStrs = [gameInstrStr, controlInstruction + " to move the ball.", nextInstrStr];
  let pauseInstStrs = [nextInstrStr];

  if (gameMode == PAUSE_MODE) {
    text("PAUSED", width / 2, firstLineY);
    
    showInstructions(pauseInstStrs, thirdLineY);
  } else if (gameMode == STARTING_MODE) {
    text("GAME COMPLETE", width / 2, firstLineY);

    textSize(32);
    text("Game " + (E.block_index+1).toFixed(0) + " of " + Object.keys(E.block_configs).length.toFixed(0), width / 2, secondLineY + 0);
  } else if (gameMode == READY_MODE) {
    if (trial_block.block_count === 0) {
      text("Welcome!", width / 2, firstLineY);
      showInstructions(welcomeInstrStrs, secondLineY);
    } else {
      text("Great job!", width / 2, firstLineY);
      textSize(32);
      text("Game " + (E.block_index+1).toFixed(0) + " of " + Object.keys(E.block_configs).length.toFixed(0), width / 2, secondLineY + 0);

      // Check for any pre-block instructions
      let blockInstrs = trial_block.block_config.params.pre_instructions;
      if (!blockInstrs) {
        blockInstrs = [];
      }
      showInstructions(blockInstrs.concat(pauseInstStrs), thirdLineY);
    }
    textSize(32);
  } else if (gameMode == COMPLETE_MODE) {
    text("EXPERIMENT COMPLETE", width / 2, firstLineY);
    textSize(32);
    text("Thank you!", width / 2, secondLineY + 0);
    redirectAtStudyCompletion(E.params.redirectUrl);
    
  } else {
    console.log("Invalid gameMode");
  }

  if (gameMode != COMPLETE_MODE) {
    if (E.params.debug) {
      text("'N' for next game", width / 2, secondLineY + 80);
      text("'R' to restart current game", width / 2, secondLineY + 120);
      text("'S' to save game data", width / 2, secondLineY + 160);
    }
  }
}

function redirectAtStudyCompletion(url) {
  if (!url) { return; }
  setTimeout(() => {
    window.location.href = url;
  }, 1000); // redirect in 1000ms = 1 second
}

function checkUserButtonPresses() {
  let eventMsg;
  if (gameMode == PLAY_MODE) {
    if (user.pause) {
      // pause game
      eventMsg = 'pause';
      gameMode = PAUSE_MODE;
      // save experiment to json
      wsLogger.saveJson(E);
      if (trial_block !== undefined) trial_block.log_pause_start();
    }
  } else if (user.pause && gameMode != COMPLETE_MODE) {
    // unpause game
    eventMsg = 'unpause';
    gameMode = PLAY_MODE;
    if (trial_block !== undefined) {
      trial_block.log_pause_end();
    }
  } else if (!E.params.isCloudStudy) {
    // the following are unavailable controls in a cloud study
    if (user.next_block && gameMode != COMPLETE_MODE) {
      // go to the next block
      eventMsg = 'new game (going to next block)';
      newGame(false);
    } else if (user.back_block && gameMode != COMPLETE_MODE) {
      // go back a block
      eventMsg = 'new game (going back a block)';
      newGame(false, true);
    } else if (user.restart_block) {
      eventMsg = 'restart block';
      newGame(true);
    }
  } else if (user.save) {
    wsLogger.saveJson(E);
  }
  if (eventMsg !== undefined) {
    wsLogger.log("interaction", {eventMsg});
  }
}

// for discrete events that we want to timestamp
function markEvent() {
  if (!E.params.isCloudStudy) {
    photodiode.trigger(50);
    clickSound.play();
  }
}

// hook up to universal controls
function keyPressed(event) { controls.keyPressed(event); }
function keyReleased(event) { controls.keyReleased(event); }
function mousePressed(event) { controls.mousePressed(event); }
function mouseReleased(event) { controls.mouseReleased(event); }

function getGameInfo() {
  return {
    width: width,
    height: height,
    levelWidth: levelWidth,
    levelHeight: levelHeight,
    levelStartX: levelStartX,
    levelEndX: levelEndX,
    ballRadius: ball.r,
    ballAccel: ballAccel,
    gravity: gravity,
    initScrollSpeed: initScrollSpeed,
    maxScrollSpeed: maxScrollSpeed,
    levelSpacing: levelSpacing,
  };
}

