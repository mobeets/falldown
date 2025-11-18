// ======================
// Global settings
// ======================
let scrollSpeed = 2.5;    // upward speed of world
let gravity = 0.5;
let ballAccel = 0.4;      // acceleration added by pressing key
let K = 8;                // number of segments per level
let levelWidth;           // total width in pixels
let levelSpacing;   // vertical distance between levels
let nLevelsVisible = 7;
let levelHeight = 10;
let cameraY = 0;
let cameraMode = 0; // options: 0 = 'follow', 1 = 'drift'
let FPS = 60;

let ball;
let levels = [];
let isPaused = false;
let isGameOver = false;
let trials = [];
let levelIndex = 0;
let gameIndex = 0;
let startTime;

function setup() {
  let windowSize = min(windowWidth, windowHeight);
  let cnv = createCanvas(windowSize, windowSize);
  cnv.parent('canvas-container'); // attach to the centered div
  levelWidth = width;

  // adjust gravity and ballAccel relative to 600x600 window
  gravity *= (width / 600);
  ballAccel *= (height / 600);

  // set level spacing so that the same number of levels are visible
  levelSpacing = width / nLevelsVisible;

  let gapSize = windowSize / K;
  ball = new Ball(width/2, 100, 0.2*gapSize);
  initGame();
}

function initGame() {
  // Set ball position
  ball.x = width/2;
  ball.y = 100;
  cameraY = 0;
  
  // Create initial levels
  levels = [];
  levelIndex = 0;
  for (let i = 0; i < 10; i++) {
    let holes = randomHoles(K);
    let y = height + i * levelSpacing;
    levelIndex++;
    levels.push(new Level(levelIndex, K, levelWidth, holes, y));
  }  
  
  isGameOver = false;
  startTime = millis();
  gameIndex++;
}

function draw() {
  frameRate(FPS);
  background(40);
  
  if (keyIsDown(LEFT_ARROW)) ball.vx -= ballAccel;
  if (keyIsDown(RIGHT_ARROW)) ball.vx += ballAccel;
  ball.vx = constrain(ball.vx, -15*ballAccel, 15*ballAccel);

  // Update ball
  let doUpdate = !isPaused && !isGameOver;
  if (doUpdate) ball.update();
  
  // Set y offset based on camera mode
  if (cameraMode === 0) {
    cameraY = ball.y - height/2;  // keep ball halfway up screen
  } else if (doUpdate) {
    cameraY += scrollSpeed;
  }
  
  // Update and render levels
  for (let lvl of levels) {
    if (doUpdate) lvl.update();
    lvl.render();
    lvl.collidesWith(ball);
    if (lvl.passedThrough(ball)) updateTrials(lvl);
  }

  // Remove levels that went off top and add new ones at bottom
  if (levels[0].y - cameraY < -50) {
    levels.shift();
    let holes = randomHoles(K);
    let newY = levels[levels.length - 1].y + levelSpacing;
    levelIndex++;
    levels.push(new Level(levelIndex, K, levelWidth, holes, newY));
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
  }
}

function keyPressed() {
  if (key === 'p') isPaused = !isPaused;
  if (key === 'n' && (isPaused || isGameOver)) initGame();
  if (key === 'm' && (isPaused || isGameOver)) cameraMode = int(!cameraMode); // toggle
  if (key === 's' && (isPaused || isGameOver)) saveTrials();
}

function randomHoles(K) {
  // choose 1–3 random holes
  let num = floor(random(1, 3));
  let arr = [];
  while (arr.length < num) {
    let idx = floor(random(K));
    if (!arr.includes(idx)) arr.push(idx);
  }
  return arr;
}

// ======================
// Ball class
// ======================
class Ball {
  constructor(x, y, r) {
    this.x = x;
    this.y = y;
    this.xprev = x;
    this.yprev = y;
    this.r = r;
    this.vx = 0;
    this.vy = 0;
  }

  update() {
    // save previous position
    this.xprev = this.x;
    this.yprev = this.y;
    this.vy += gravity;
    this.x += this.vx;
    this.y += this.vy;
    
    // Slow horizontal motion (friction)
    this.vx *= 0.95;

    // Keep inside screen
    if (this.x < this.r) { this.x = this.r; this.vx = 0; }
    if (this.x > width - this.r) { this.x = width - this.r; this.vx = 0; }
  }

  render() {
    fill(255, 200, 0);
    noStroke();
    circle(this.x, this.y - cameraY, this.r * 2);
  }
}


// ======================
// RectSegment class
// ======================
class RectSegment {
  constructor(index, x, y, w) {
    this.index = index;
    this.x = x;
    this.y = y;
    this.w = w;
    this.h = levelHeight;
  }

  update() {
    // if (cameraMode === 1) this.y -= scrollSpeed;
  }

  render() {
    fill(180);
    rect(this.x, this.y - cameraY, this.w, this.h);
  }

  collidesWith(ball) {
    const hit = collideRectCircle(this.x, this.y, this.w, this.h, ball.x, ball.y, ball.r * 2);
    if (hit.colliding) {
      let prevBallX = ball.x;
      let prevBallY = ball.y;
      
      if (hit.yEdge === 0) {
        // ball is hitting side of platform
        if (hit.xEdge === -1) ball.x = this.x - ball.r;
        else if (hit.xEdge === 1) ball.x = this.x + this.w + ball.r;
        ball.vx = 0;
      }
      if (hit.xEdge === 0) {
        // ball is hitting top of platform
        ball.y = this.y - ball.r;
        ball.vy = min(ball.vy, 0);
      }
    }
    return hit.colliding;
  }
}

collideRectCircle = function (rx, ry, rw, rh, cx, cy, diameter) {
  // temporary variables to set edges for testing
  var testX = cx;
  var testY = cy;
  let xEdge = 0;
  let yEdge = 0;

  // which edge is closest?
  if (cx < rx){ testX = rx; xEdge = -1; // left edge
  }else if (cx > rx+rw){ testX = rx+rw; xEdge = 1; }   // right edge

  if (cy < ry){ testY = ry; yEdge = -1; // top edge
  }else if (cy > ry+rh){ testY = ry+rh; yEdge = 1; }   // bottom edge

  // // get distance from closest edges
  var distance = this.dist(cx,cy,testX,testY)

  // if the distance is less than the radius, collision!
  if (distance <= diameter/2) {
    return {colliding: true, xEdge: xEdge, yEdge: yEdge};
  }
  return {colliding: false, xEdge: xEdge, yEdge: yEdge};
};

// ======================
// Level class
// ======================
class Level {
  constructor(index, K, width, holes, y) {
    this.index = index;
    this.K = K;
    this.width = width;
    this.holes = holes;
    this.y = y;
    this.holeUsed = -1;
    this.ballTouched = false;
    
    this.segments = [];
    let segW = width / K;

    this.segmentExists = [];
    for (let i = 0; i < K; i++) {
      if (!holes.includes(i)) {
        this.segmentExists.push(1);
        this.segments.push(new RectSegment(i, i * segW, y, segW));
      } else {
        this.segmentExists.push(0);
      }
    }
  }

  update() {
    // if (cameraMode === 1) this.y -= scrollSpeed;
    for (let seg of this.segments) {
      seg.update();
    }
  }

  render() {
    for (let seg of this.segments) {
      seg.render();
    }
    // textSize(12);
    // text(this.holeUsed.toString(), width/2, this.y - cameraY);
  }

  collidesWith(ball) {
    for (let seg of this.segments) {      
      let didTouch = seg.collidesWith(ball);
      this.ballTouched = this.ballTouched || didTouch;
    }
  }
  
  passedThrough(ball) {
    if (this.holeUsed > -1) return false;
    
    // Check if the ball has crossed the level vertically
    if (ball.y - 2*ball.r > this.y + levelHeight) {

      // Compute horizontal grid index of the ball
      let segWidth = this.width / this.K;
      this.holeUsed = floor(ball.x / segWidth);

      // Verify that this grid index is actually a hole
      if (!this.holes.includes(this.holeUsed)) {
        // Ball didn’t pass through a hole, ignore
        this.holeUsed = -1;
      }
      return this.holeUsed > -1;
    }
    return this.holeUsed > -1;
  }
  
  toJSON() {
    return {
      index: this.index,
      levelY: this.y,
      // represent level as a K-bit integer
      levelInt: bitsToByte(this.segmentExists, this.K), 
      holeUsed: this.holeUsed,
      ballTouched: this.ballTouched,
    };
  }
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
  trial.time = millis() - startTime;
  trial.gameIndex = gameIndex;
  trial.cameraMode = cameraMode;
  trial.ballX = ball.x;
  trial.ballY = ball.y;
  trial.cameraY = cameraY;
  trials.push(trial);
}

function saveTrials() {
  let gameInfo = {
    width: width,
    height: height,
    ballRadius: ball.r,
    ballAccel: ballAccel,
    gravity: gravity,
    levelHeight: levelHeight,
    levelSpacing: levelSpacing,
    segmentsPerLevel: K,
    scrollSpeed: scrollSpeed,
    FPS: FPS,
  };

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
