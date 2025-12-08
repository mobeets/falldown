// ======================
// Global settings
// ======================
let scrollSpeed = 2.2;    // upward speed of world
let gravity = 0.5;
let ballAccel = 0.4;      // acceleration added by pressing key
let friction = 0.95; // decay on ball's vx
let K = 12;                // number of segments per level
let levelWidth;           // total width in pixels
let levelSpacing;   // vertical distance between levels
let nLevelsVisible = 7;
let levelHeight = 10;
let cameraY = 0;
let cameraMode = 0; // options: 0 = 'follow', 1 = 'drift'
let FPS = 60;
// let modeSwitchRates = [0.05, 0.1];
let modeSwitchRates = [0.0, 0.1];
let minLevelsPerMode = 10;
let modeSwitchCooldown = 0;
// let modeRectColors = ['gray', 'red'];
let modeRectColors = ['gray', 'gray'];

// todo: track pause times

let ball;
let levels = [];
let isPaused = false;
let isGameOver = false;
let trials = [];
let levelIndex = 0;
let gameIndex = 0;
let startTime;
let planningDepth = 2;
let holePlanner;

function setup() {
  let windowSize = min(windowWidth, windowHeight);
  let cnv = createCanvas(windowSize, windowSize);
  cnv.parent('canvas-container'); // attach to the centered div
  levelWidth = width;

  // adjust gravity and ballAccel relative to 600x600 window
  gravity *= 1.5 * (width / 600);
  ballAccel *= (height / 600);

  // set level spacing so that the same number of levels are visible
  levelSpacing = width / nLevelsVisible;

  let gapSize = windowSize / K;
  ball = new Ball(width/2, 100, 0.1*gapSize);
  initGame();
}

function initGame() {
  // Set ball position
  ball.x = width/2;
  ball.y = 100;
  cameraY = 0;

  holePlanner = new HolePlanner(K, planningDepth);
  
  // Create initial levels
  levels = [];
  levelIndex = 0;
  for (let i = 0; i < 10; i++) {
    let holes = holePlanner.next_holes();
    let y = height + i * levelSpacing;
    levelIndex++;

    levels.push(new Level(levelIndex, K, levelWidth, holes, y, cameraMode, false));
  }  
  
  isGameOver = false;
  startTime = millis();
  gameIndex++;
}

function checkForModeSwitch(modeIndex, modeSwitchRate) {
  let doModeSwitch = false;
  if (modeSwitchCooldown > 0) {
    modeSwitchCooldown--;
  } else {
    doModeSwitch = random() < modeSwitchRate;
    if (doModeSwitch) modeSwitchCooldown = minLevelsPerMode;
  }
  if (doModeSwitch) modeIndex = int(!modeIndex);
  return {modeIndex, doModeSwitch};
}

function draw() {
  frameRate(FPS);
  background(40);
  
  // Update ball
  let doUpdate = !isPaused && !isGameOver;

  if (doUpdate) {
    if (keyIsDown(LEFT_ARROW)) ball.vx -= ballAccel;
    if (keyIsDown(RIGHT_ARROW)) ball.vx += ballAccel;
    ball.vx = constrain(ball.vx, -15*ballAccel, 15*ballAccel);
    ball.update();
  }
  
  // Set y offset based on camera mode
  if (cameraMode === 0) {
    // keep ball halfway up screen, but smooth movements
    cameraY = 0.25*(ball.y - height/2) + 0.75*cameraY;
  } else if (doUpdate) {
    cameraY += scrollSpeed;
  }
  
  // Update and render levels
  for (let lvl of levels) {
    if (doUpdate) lvl.update();
    lvl.render();
    lvl.collidesWith(ball);
    if (lvl.passedThrough(ball)) {
      updateTrials(lvl);
      // toggle mode when we pass through
      if (lvl.isModeSwitch) cameraMode = lvl.modeIndex; //int(!cameraMode);
    }
  }

  // Remove levels that went off top and add new ones at bottom
  if (levels[0].y - cameraY < -50) {
    levels.shift();
    
    // Set params for new level
    levelIndex++;
    let holes = holePlanner.next_holes();
    let newY = levels[levels.length - 1].y + levelSpacing;
    let modeIndex = levels[levels.length - 1].modeIndex;

    // Check for mode switch on this level
    // let modeInfo = checkForModeSwitch(modeIndex, modeSwitchRates[cameraMode]);
    let modeInfo = checkForModeSwitch(modeIndex, modeSwitchRates[modeIndex]);
    
    // Create new level
    levels.push(new Level(levelIndex, K, levelWidth, holes, newY, modeInfo.modeIndex, modeInfo.doModeSwitch));
  }
  
  // Render ball
  ball.render();

  // Game over condition
  if ((cameraMode === 1) && (ball.y - cameraY < 0)) {
    isGameOver = true;
  }
  
  if (!doUpdate) {
    textAlign(CENTER, CENTER);
    fill(color(50, 50, 50, 200));
    rect(0, 0, width, height);
    fill(255);
    textSize(48);
    if (isGameOver) {
      text("GAME OVER", width/2, height/2);
    } else if (isPaused) {
      text("PAUSED", width/2, height/2);
    }
    if (cameraMode === 1) {
      textSize(24);
      text("Scroll speed: " + scrollSpeed.toFixed(2), width/2, height/2 + 60);
    }
  }
}

function keyPressed() {
  if (key === 'p') isPaused = !isPaused;
  if (key === 'n' && (isPaused || isGameOver)) initGame();
  // if (key === 'm' && (isPaused || isGameOver)) cameraMode = int(!cameraMode); // toggle
  if (key === 's' && (isPaused || isGameOver)) saveTrials();
  if (keyCode === UP_ARROW && (isPaused || isGameOver)) scrollSpeed += 0.1;
  if (keyCode === DOWN_ARROW && (isPaused || isGameOver)) scrollSpeed -= 0.1;
}

function bitsToByte(bits, K) {
  let value = 0;
  for (let i = 0; i < K; i++) {
    value = (value << 1) | bits[i];
  }
  return value;
}

function updateTrials(level) {
  let trial = level.toJSON();
  trial.timePassedThru = millis() - startTime;
  trial.gameIndex = gameIndex;
  trial.cameraMode = cameraMode;
  trial.scrollSpeed = scrollSpeed;
  trial.ballX = ball.x;
  trial.ballY = ball.y;
  trial.cameraY = cameraY;
  trials.push(trial);
}

function getGameInfo() {
  return {
    width: width,
    height: height,
    ballRadius: ball.r,
    ballAccel: ballAccel,
    gravity: gravity,
    levelHeight: levelHeight,
    levelSpacing: levelSpacing,
    modeSwitchRates: modeSwitchRates,
    minLevelsPerMode: minLevelsPerMode,
    segmentsPerLevel: K,
    scrollSpeed: scrollSpeed,
    FPS: FPS,
  };
}

function saveTrials() {
  let gameInfo = getGameInfo();
  let jsonString = JSON.stringify({gameInfo: gameInfo,
    trials: trials}, null, 2); // Pretty-print with 2-space indent

  // Create a Blob from the JSON string
  let blob = new Blob([jsonString], { type: 'application/json' });

  // Create a temporary download link
  let url = URL.createObjectURL(blob);
  let a = document.createElement('a');
  a.href = url;
  a.download = 'data.json';
  a.click();

  // Clean up the URL object
  URL.revokeObjectURL(url);
}
