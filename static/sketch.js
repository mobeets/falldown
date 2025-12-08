// ======================
// Global settings
// ======================
let gravity = 0.5;
let ballAccel = 0.4;      // acceleration added by pressing key
let cameraMode = 0; // options: 0 = 'follow', 1 = 'drift'

// todo: track pause times

let levels = [];
let isPaused = false;
let isGameOver = false;
let levelIndex = 0;
let cameraY = 0;
let planningDepth = 2;

let modeSwitchCooldown; // min levels per cameraMode
let levelWidth;     // total width in pixels
let levelSpacing;   // vertical distance between levels
let ball;
let experiment;
let trial_block;
let holePlanner;
let config;
let photodiode;
let controls;
let user;
let clickSound;

function preload() {
  clickSound = new Audio('static/click.mp3');
  config = loadConfig();
}

function setup() {
  let windowSize = min(windowWidth, windowHeight);
  let cnv = createCanvas(windowSize, windowSize);
  cnv.parent('canvas-container'); // attach to the centered div
  levelWidth = width;

  E = new Experiment(config);
  
  photodiode = new Photodiode(E.params.photodiode, width, height);
  controls = new UnifiedControls(wsLogger);
  user = new TaskControls(controls);

  // adjust gravity and ballAccel relative to 600x600 window
  gravity *= 1.5 * (width / 600);
  ballAccel *= (height / 600);

  // set level spacing so that the same number of levels are visible
  levelSpacing = width / E.params.nLevelsVisible;

  let gapSize = windowSize / E.params.nSegments;
  ball = new Ball(width/2, 100, 0.1*gapSize);
  initGame();
}

function initGame() {
  // Set ball position
  ball.x = width/2;
  ball.y = 100;
  cameraY = 0;
  modeSwitchCooldown = E.params.minLevelsPerMode;

  holePlanner = new HolePlanner(E.params.nSegments, planningDepth);
  
  // Create initial levels
  levels = [];
  levelIndex = 0;
  for (let i = 0; i < 10; i++) {
    let holes = holePlanner.next_holes();
    let y = height + i * levelSpacing;
    levelIndex++;

    levels.push(new Level(levelIndex, E.params.nSegments, levelWidth, E.params.levelHeight, holes, y, cameraMode, E.params.modeRectColors[cameraMode], false));
  }
  
  isGameOver = false;
  trial_block = E.new_block();
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

function draw() {
  frameRate(E.params.FPS);
  background(40);
  controls.update();
  checkUserButtonPresses();
  
  // Update ball
  let doUpdate = !isPaused && !isGameOver;

  if (doUpdate) {
    if (user.moveLeft) ball.vx -= ballAccel;
    if (user.moveRight) ball.vx += ballAccel;
    ball.vx = constrain(ball.vx, -15*ballAccel, 15*ballAccel);
    ball.update();
  }
  
  // Set y offset based on camera mode
  if (cameraMode === 0) {
    // keep ball halfway up screen, but smooth movements
    cameraY = 0.25*(ball.y - height/2) + 0.75*cameraY;
  } else if (doUpdate) {
    cameraY += E.params.scrollSpeed;
  }
  
  // Update and render levels
  for (let lvl of levels) {
    if (doUpdate) lvl.update();
    lvl.render();
    lvl.collidesWith(ball);
    if (lvl.passedThrough(ball)) {
      trial_block.add_trial(lvl);
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
    let holes = holePlanner.next_holes();
    let newY = levels[levels.length - 1].y + levelSpacing;
    let modeIndex = levels[levels.length - 1].modeIndex;

    // Check for mode switch on this level
    let modeInfo = checkForModeSwitch(modeIndex, E.params.modeSwitchRates[modeIndex]);
    
    // Create new level
    levels.push(new Level(levelIndex, E.params.nSegments, levelWidth, E.params.levelHeight, holes, newY, modeInfo.modeIndex, E.params.modeRectColors[modeInfo.modeIndex], modeInfo.doModeSwitch));
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
      text("Scroll speed: " + E.params.scrollSpeed.toFixed(2), width/2, height/2 + 60);
    }
  }

  // render photodiode last
  photodiode.update();
  photodiode.render();
}

function checkUserButtonPresses() {
  if (user.pause) isPaused = !isPaused;
  if (user.save && (isPaused || isGameOver)) manuallySaveToJSON(E);
}

// for discrete events that we want to timestamp
function markEvent() {
  photodiode.trigger(50);
  clickSound.play();
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
    ballRadius: ball.r,
    ballAccel: ballAccel,
    gravity: gravity,
    levelSpacing: levelSpacing,
  };
}

