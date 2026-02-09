// ======================
// HolePlanner class
// ======================


class HolePlanner {
  constructor(nSegments, trialsData) {
    this.nSegments = nSegments;
    
    // --- ROBUST DATA LOADING ---
    // 1. If it's an array (List of trials), use it directly.
    if (Array.isArray(trialsData)) {
       this.trials = trialsData;
    } 
    // 2. If it's a single object that HAS a 'levels' property (The structure you pasted),
    //    we wrap it in an array so the rest of the code works.
    else if (trialsData && trialsData.levels) {
       this.trials = [trialsData];
    }
    // 3. Fallback: It might be an object of numbered keys { "0": {...}, "1": {...} }
    else if (trialsData) {
       this.trials = Object.values(trialsData);
    } 
    else {
       this.trials = []; // Empty fallback to prevent crash
       console.error("HolePlanner: No valid data found!");
    }

    this.trialIndex = 0;
    this.levelIndex = 0;
  }

  next_holes() {
    // Safety check: Return empty holes if data is missing, don't crash
    if (!this.trials || this.trials.length === 0) return { hole_locations: [] };

    // Get current trial
    let currentTrial = this.trials[this.trialIndex];

    // Safety check: If this trial is broken/empty, skip or return empty
    if (!currentTrial || !currentTrial.levels) return { hole_locations: [] };

    // Get current level
    let currentLevelObj = currentTrial.levels[this.levelIndex];
    let rawHoles = currentLevelObj.holes || []; // Default to empty array if missing

    // Convert floats to segment indices
    let processedHoles = rawHoles.map(val => {
        let clamped = Math.max(0, Math.min(1, val));
        return Math.floor(clamped * this.nSegments);
    });

    // Advance indices
    this.levelIndex++;

    // If we finished the levels in this trial...
    if (this.levelIndex >= currentTrial.levels.length) {
      this.levelIndex = 0;      
      this.trialIndex++;        
      
      // If we finished ALL trials, loop back to start
      if (this.trialIndex >= this.trials.length) {
        this.trialIndex = 0;
      }
    }

    return {
      plan_index: this.trialIndex,
      layer_index: this.levelIndex, // Note: this is just for logging, doesn't affect gameplay
      hole_locations: processedHoles
    };
  }
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
    this.vx *= E.params.friction;

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
  constructor(index, x, y, w, h, color) {
    this.index = index;
    this.x = x;
    this.y = y;
    this.w = w;
    this.h = h;
    this.color = color;
  }

  update() {
  }

  render() {
    fill(this.color);
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
  constructor(index, nSegments, width, height, holes, y, modeIndex, color, isModeSwitch) {
    this.index = index;
    this.nSegments = nSegments;
    this.width = width;
    this.height = height;
    this.holes = holes;
    this.hole_locations = this.holes.hole_locations;
    this.y = y;
    this.holeUsed = -1;
    this.ballTouched = false;
    this.modeIndex = modeIndex;
    this.color = color;
    this.isModeSwitch = isModeSwitch;
    
    this.segments = [];
    let segW = width / this.nSegments;

    this.segmentExists = [];
    for (let i = 0; i < this.nSegments; i++) {
      if (!this.hole_locations.includes(i)) {
        this.segmentExists.push(1);
        this.segments.push(new RectSegment(i, i * segW, y, segW, this.height, this.color));
      } else {
        this.segmentExists.push(0);
      }
    }
  }

  update() {
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
    if (ball.y - 2*ball.r > this.y + this.height) {

      // Compute horizontal grid index of the ball
      let segWidth = this.width / this.nSegments;
      this.holeUsed = floor(ball.x / segWidth);

      // Verify that this grid index is actually a hole
      if (!this.hole_locations.includes(this.holeUsed)) {
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
      // levelInt: bitsToByte(this.segmentExists, this.K), 
      holeUsed: this.holeUsed,
      ballTouched: this.ballTouched,
      modeIndex: this.modeIndex,
      holes: this.holes,
      isModeSwitch: this.isModeSwitch,
    };
  }
}
