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
    if (this.x - levelStartX < this.r) { this.x = levelStartX + this.r; this.vx = 0; }
    if (this.x - levelStartX > levelWidth - this.r) { this.x = levelStartX + levelWidth - this.r; this.vx = 0; }
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
  constructor(index, nSegments, width, height, trial, x, y, modeIndex, color, isModeSwitch) {
    this.index = index;
    this.nSegments = nSegments;
    this.width = width;
    this.height = height;
    this.trial = trial;
    this.hole_locations = this.trial.hole_locations;
    this.x = x;
    this.y = y;
    this.holeUsed = -1;
    this.ballTouched = false;
    this.modeIndex = modeIndex;
    this.color = color;
    this.isModeSwitch = isModeSwitch;
    this.timeBallFirstTouched;
    
    this.segments = [];
    let segW = this.width / this.nSegments;

    this.segmentExists = [];
    for (let i = 0; i < this.nSegments; i++) {
      if (!this.hole_locations.includes(i)) {
        this.segmentExists.push(1);
        this.segments.push(new RectSegment(i, this.x + i * segW, y, segW, this.height, this.color));
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
  }

  collidesWith(ball) {
    for (let seg of this.segments) {      
      let didTouch = seg.collidesWith(ball);
      if (didTouch && !this.ballTouched) {
        this.timeBallFirstTouched = performance.now();
      }
      this.ballTouched = this.ballTouched || didTouch;
    }
  }
  
  passedThrough(ball) {
    if (this.holeUsed > -1) return false;
    if (!this.hole_locations.length) return false;
    
    // Check if the ball has crossed the level vertically
    if (ball.y - 2*ball.r > this.y + this.height) {

      // Compute horizontal grid index of the ball
      let segWidth = this.width / this.nSegments;

      // Compute which segment the ball was just in based on its previous x position
      let ballSegment = floor((ball.xprev - this.x) / segWidth);

      // find nearest hole location to ballSegment
      this.holeUsed = this.hole_locations.reduce((nearest, hole) =>
        abs(hole - ballSegment) < abs(nearest - ballSegment) ? hole : nearest
      );
      return true;
    }
    return false;
  }
  
  toJSON() {
    return {
      index: this.index,
      levelY: this.y,
      modeIndex: this.modeIndex,
      isModeSwitch: this.isModeSwitch,
      holeUsed: this.holeUsed,
      ballTouched: this.ballTouched,
      timeBallFirstTouched: this.timeBallFirstTouched
    };
  }
}
