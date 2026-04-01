let levels = [];
let levelIndex = 0;
let cameraY = 0;
let cameraYTarget = 0;
let planningDepth = 2;
let cameraMode; // options: 0 = 'follow', 1 = 'drift'
let cloudResearchRedirect = 'https://example.com/';

let gravity;
let ballAccel;      // acceleration added by pressing key
let modeSwitchCooldown; // min levels per cameraMode
let levelWidth;     // total width in pixels
let levelStartX;    // x value where levels start (to center them)
let levelEndX;      // x value where levels end
let levelSpacing;   // vertical distance between levels
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
  levelStartX = (windowWidth - levelWidth) / 2;
  levelEndX = levelStartX + levelWidth;
  
  cameraMode = E.params.startCameraMode;

  photodiode = new Photodiode(E.params.photodiode, width, height);
  controls = new UnifiedControls(wsLogger);
  user = new TaskControls(controls);

  // adjust gravity and ballAccel relative to 600x600 window
  gravity = E.params.relativeGravity * (levelWidth / 600);
  ballAccel = E.params.relativeBallAccel * (height / 600);

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
  ball.x = width/2;
  ball.y = 100;
  cameraY = 0;
  modeSwitchCooldown = E.params.minLevelsPerMode;
  
  // Create initial levels
  levels = [];
  levelIndex = 0;
  let prevTrial;
  for (let i = 0; i < 10; i++) {
    trial = trial_block.next_trial();

    let y = height/2 + i * levelSpacing;
    levelIndex++;

    levels.push(new Level(levelIndex, E.params.nSegments, levelWidth, E.params.levelHeight, trial, levelStartX, y, cameraMode, E.params.modeRectColors[cameraMode], false));
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
      ball.vx -= ballAccel;
      E.log_user_input(-1);
    }
    if (user.moveRight) {
      ball.vx += ballAccel;
      E.log_user_input(1);
    }
    ball.vx = constrain(ball.vx, -15*ballAccel, 15*ballAccel);
    E.log_states(ball); // logs ball and camera states
    ball.update();

    // Set y offset based on camera mode
    if (cameraMode === 0) {
      // keep ball halfway up screen, but smooth movements
      cameraY = 0.25*(ball.y - height/2) + 0.75*cameraY;
      cameraYTarget = cameraY;
    } else if (cameraMode === 1) {
      cameraY += E.params.scrollSpeed;
    } else {
      // if ball is in lower part of screen, keep camera fixed on ball so that ball can't go lower, but continue to update cameraYTarget so that when ball goes back up it will be back in the right place
      if (ball.y - cameraYTarget > 3*height/5) {
        cameraY = 0.1*(ball.y - 3*height/5) + 0.9*cameraY;
        cameraYTarget = cameraY;
      } else {
        cameraY = cameraYTarget;
      }      
      cameraYTarget += E.params.scrollSpeed;
    }
    
    // Update and render levels
    for (let lvl of levels) {
      lvl.update();
      lvl.render();
      lvl.collidesWith(ball);
      if (lvl.passedThrough(ball)) {
        // trial_block.add_trial(lvl);
        lvl.trial.trigger(decisionEvent(lvl));
        markEvent(); // trigger photodiode and play sound
        // toggle mode when we pass through
        if (lvl.isModeSwitch) cameraMode = lvl.modeIndex; //int(!cameraMode);
      }
    }
  
    // Remove levels that went off top and add new ones at bottom
    if (levels[0].y - cameraY < -50) {
      levels.shift();
      
      // Set params for new level
      levelIndex++;
      let trial = trial_block.next_trial();
      if (trial === undefined) {
          newGame(false);
      } else {
        let newY = levels[levels.length - 1].y + levelSpacing;
        let modeIndex = levels[levels.length - 1].modeIndex;
  
        // Check for mode switch on this level
        let modeInfo = checkForModeSwitch(modeIndex, E.params.modeSwitchRates[modeIndex]);
        
        // Create new level
        levels.push(new Level(levelIndex, E.params.nSegments, levelWidth, E.params.levelHeight, trial, levelStartX, newY, modeInfo.modeIndex, E.params.modeRectColors[modeInfo.modeIndex], modeInfo.doModeSwitch));
      }
    }
    // Render ball
    ball.render();
    
    // Game over condition
    if ((cameraMode === 1) && (ball.y - cameraY < 0)) {
      newGame(false);
    }
  } else {
    drawPauseScreen();
  }
  
  // render photodiode last
  photodiode.update();
  photodiode.render();
}


function showText(text_lines, yOffset) {

  textSize(20);
  textFont('arial');
  fill('white');
  for (var i = 0; i < text_lines.length; i++) {
    text(text_lines[i], width / 2, yOffset + 40*i);
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

  if (gameMode == PAUSE_MODE) {
    text("PAUSED", width / 2, firstLineY);
    // if (trial_block.instructions) {
    //   showInstructions(secondLineY + 100);
    //   showImages(secondLineY + 300);
    // }
    if (E.params.isCloudStudy) {
      showText(["Press spacebar to continue."], thirdLineY);
    }
  } else if (gameMode == STARTING_MODE) {
    text("GAME COMPLETE", width / 2, firstLineY);

    // fill('black');
    textSize(32);
    text("Game " + (E.block_index+1).toFixed(0) + " of " + Object.keys(E.block_configs).length.toFixed(0), width / 2, secondLineY + 0);
    // text("Score: " + trial_block.score.toFixed(0) + " out of " + trial_block.trials.length, width / 2, secondLineY + 40);
  } else if (gameMode == READY_MODE) {
    if (trial_block.block_count === 0) {
      text("Welcome!", width / 2, firstLineY);
    } else {
      text("Great job!", width / 2, firstLineY);
      if (lastScore !== undefined) {
        textSize(24);
        text(`Completed ${lastScore}`, width / 2, firstLineY + 50);
      }
    }
    // fill('black');
    textSize(32);
    text("Game " + (E.block_index+1).toFixed(0) + " of " + Object.keys(E.block_configs).length.toFixed(0), width / 2, secondLineY + 0);
    // if (trial_block.is_practice) {
    //   fill('#9e442f');
    //   text("Practice round!", width / 2, secondLineY + 40);
    // }
    // if (trial_block.instructions) {
    //   showInstructions(secondLineY + 100);
    //   showImages(secondLineY + 300);
    //   showJet();
    // }
    if (E.params.isCloudStudy) {
      showText(["Press spacebar to continue."], thirdLineY);
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
    }
  } else if (user.pause && gameMode != COMPLETE_MODE) {
    // unpause game
    eventMsg = 'unpause';
    gameMode = PLAY_MODE;
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
    levelStartX: levelStartX,
    levelEndX: levelEndX,
    ballRadius: ball.r,
    ballAccel: ballAccel,
    gravity: gravity,
    levelSpacing: levelSpacing,
  };
}

