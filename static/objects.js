// ======================
// HolePlanner class
// ======================
class HolePlanner {
  constructor(nSegments, planningDepth) {
    this.nSegments = nSegments;
    this.planningDepth = planningDepth;
    this.holes_queued = [];
    this.plan_index = 0;
    this.plan_next_chunk();
  }

  random_hole_locations(nSegments, nHoles, skipEdges = false, holesToAvoid) {
    // creates array with length nHoles, where indices correspond to hole locations
    let arr = [];
    if (holesToAvoid === undefined) { holesToAvoid = []; }
    while (arr.length < nHoles) {
      let idx;
      if (skipEdges) {
        // don't allow holes on left-most or right-most edge
        idx = floor(random(0, nSegments-2))+1; // 1...nSegments-2
      } else {
        idx = floor(random(0, nSegments)); // 0...nSegments-1
      }
      // we will add this hole if it isn't already present,
      // and if it is not in our list of holesToAvoid
      if (!arr.includes(idx) && !holesToAvoid.includes(idx)) {
        arr.push(idx);
      }
    }
    return arr;
  }

  random_holes_flanking(nSegments, centerHole) {
    // returns two holes, one to the left of centerHole, and one to the right
    let idxL = floor(random(0, centerHole)); // 0...centerHole-1
    let idxR = floor(random(centerHole+1, nSegments)); // centerHole+1...nSegments-1
    return [idxL, idxR];
  }

  plan_next_chunk() {
    // adds to holes_queued
    // where each entry is {plan_index, layer_index, hole_locations}
    this.plan_index++;
    let all_hole_locations;
    let curPlanningDepth = floor(random(0,this.planningDepth))+1;
    if (curPlanningDepth === 1) {
      // 1-hole, 2-hole, 1-hole
      // where greedy solution is optimal (b/c 1st and 3rd layers are identical)
      let holeA = this.random_hole_locations(this.nSegments, 1, true);
      let holeB = this.random_holes_flanking(this.nSegments, holeA[0]);
      let holeC = [...holeA];
      all_hole_locations = [holeA, holeB, holeC];
    } else if (curPlanningDepth === 2) {
      // 1-hole, 2-hole, 1-hole
      // where 2-step solution is better than greedy solution
      let holeA = this.random_hole_locations(this.nSegments, 1, true);
      let holeB = this.random_holes_flanking(this.nSegments, holeA[0]);
      let holeC = this.random_hole_locations(this.nSegments, 1, false, holeB);
      let pathL1 = Math.abs(holeA[0]-holeB[0]);
      let pathR1 = Math.abs(holeA[0]-holeB[1]);
      let pathL2 = pathL1 + Math.abs(holeB[0]-holeC[0]);
      let pathR2 = pathR1 + Math.abs(holeB[1]-holeC[0]);
      // if pathL1 < pathR1, then we want pathL2 > pathR2
      // else, we want pathL2 < pathR2
      all_hole_locations = [holeA, holeB, holeC];
    } else if (curPlanningDepth === 3) {
      // 1-hole, 2-hole, 2-hole, 1-hole
      // where greedy/2-step solution are the same, 3-step solution is different
      // and 3-step solution is better than greedy/2-step
      let holeA = this.random_hole_locations(this.nSegments, 1, true);
      let holeB = this.random_holes_flanking(this.nSegments, holeA[0]);
      let holeC = this.random_hole_locations(this.nSegments, 2, false, holeB);
      let holeD = this.random_hole_locations(this.nSegments, 1, false, holeC);
      all_hole_locations = [holeA, holeB, holeC, holeD];
    } else {
      // not implemented
      return;
    }

    for (var i = 0; i < all_hole_locations.length; i++) {
      let curHole = {
        plan_index: this.plan_index,
        plan_depth: curPlanningDepth,
        layer_index: i,
        hole_locations: all_hole_locations[i]
      };
      this.holes_queued.push(curHole);
    }
  }

  plan_random_layer() {
    this.plan_index++;
    let hole_locations = this.random_hole_locations(this.nSegments, 1);
    return {plan_index: this.plan_index, plan_depth: 0, layer_index: 0, hole_locations: hole_locations};
  }

  next_holes() {
    if (this.holes_queued.length === 0) this.plan_next_chunk();
    let holes;
    if (this.holes_queued.length === 0) {
      // did not plan anything, so we choose a 1-hole layer randomly
      holes = this.plan_random_layer();
    } else {
      holes = this.holes_queued.shift();
    }
    console.log(holes);
    return holes;
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
  constructor(index, x, y, w, color) {
    this.index = index;
    this.x = x;
    this.y = y;
    this.w = w;
    this.h = levelHeight;
    this.color = color;
  }

  update() {
    // if (cameraMode === 1) this.y -= scrollSpeed;
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
  constructor(index, K, width, holes, y, modeIndex, isModeSwitch) {
    this.index = index;
    this.K = K;
    this.width = width;
    this.holes = holes;
    this.hole_locations = this.holes.hole_locations;
    this.y = y;
    this.holeUsed = -1;
    this.ballTouched = false;
    this.modeIndex = modeIndex;
    this.color = modeRectColors[this.modeIndex];
    this.isModeSwitch = isModeSwitch;
    
    this.segments = [];
    let segW = width / K;

    this.segmentExists = [];
    for (let i = 0; i < K; i++) {
      if (!this.hole_locations.includes(i)) {
        this.segmentExists.push(1);
        this.segments.push(new RectSegment(i, i * segW, y, segW, this.color));
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
      levelInt: bitsToByte(this.segmentExists, this.K), 
      holeUsed: this.holeUsed,
      ballTouched: this.ballTouched,
      modeIndex: this.modeIndex,
      holes: this.holes,
      isModeSwitch: this.isModeSwitch,
    };
  }
}
