// Run with runLandingRegression(objectsSource) in a JavaScript runtime.
function runLandingRegression(source) {
  const create = new Function('E', 'gravity', 'levelStartX', 'levelWidth', 'min', 'dist',
    source + '\nreturn {Ball, RectSegment};');
  let cases = 0;
  for (const width of [320, 600, 1200, 2048]) {
    for (const direction of [-1, 1]) {
      const gap = width / 12;
      const gravity = 0.75 * width / 600;
      const {Ball, RectSegment} = create({params:{friction:0.92}}, gravity, 0, width,
        Math.min, (a,b,c,d) => Math.hypot(a-c,b-d));
      // Last platform has its only hole at the destination edge. Solid
      // segment seams and a deep first-frame landing must not stop rolling.
      const hole = direction > 0 ? 11 : 0;
      const segments = Array.from({length:12}, (_,i) => i).filter(i => i !== hole)
        .map(i => new RectSegment(i, i*gap, 200, gap, 10, 'gray'));
      const ball = new Ball((direction > 0 ? 2.5 : 9.5)*gap, 200-0.1*gap-1, 0.1*gap);
      ball.vy = 18 * width / 600;
      const speed = 8 * 0.6 * width / 600;
      let passed = false;
      for (let frame=0; frame<1000; frame++) {
        const before = ball.x;
        ball.vx = direction * speed;
        ball.update();
        for (const segment of segments) segment.collidesWith(ball);
        if (frame === 0 && ball.vx === 0) throw Error('Landing incorrectly zeroed horizontal velocity');
        if ((ball.x-before)*direction <= 0) throw Error('Held movement stalled at width '+width);
        if (ball.y-ball.r > 210) { passed = true; break; }
      }
      if (!passed) throw Error('Could not complete final platform at width '+width);
      cases++;
    }
  }
  return cases;
}
