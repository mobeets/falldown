
class Photodiode {
  constructor(params = {}, width, height) {
    const defaults = {
      x: 0,
      y: undefined,
      size: 200,
      color: 'white',
      default_duration: 100 // milliseconds
    };
    // Merge caller params on top of defaults
    const finalParams = { ...defaults, ...params };

    this.x = finalParams.x;
    this.y = finalParams.y;
    this.size = finalParams.size;
    this.color = finalParams.color;
    this.default_duration = finalParams.default_duration;

    this.init(width, height);
    this.start_time;
    this.duration;
  }

  init(width, height) {
    // if (x,y) are undefined use (width,height) to offset
    if (this.x === undefined) {
      this.x = width - this.size;
    } else if (this.y === undefined) {
      this.y = height - this.size;
    }
  }

  trigger(duration = this.default_duration) {
    this.start_time = millis();
    this.duration = duration;
  }

  update() {
    if (this.start_time === undefined) return;
    if (millis() - this.start_time > this.duration) {
      this.start_time = undefined;
      this.duration = undefined;
    }
  }

  render() {
    if (this.start_time === undefined) return;
    rectMode(CORNER);
    noStroke();
    fill(this.color);
    square(this.x, this.y, this.size);
  }

  toJSON() {
    // outputs all of object's variables as a json object
    return Object.assign({}, this);
  }
}
