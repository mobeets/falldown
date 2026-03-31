class TaskControls {
  constructor(unified) {
    this.u = unified;
  }

  get pause() {
    return this.u.justPressedKey(' ') || this.u.justPressedKey('p') || this.u.justPressedButton("START");
  }

  get save() {
    return this.u.justPressedKey('s');
  }

  get back_block() {
    return this.u.justPressedKey('b');
  }

  get next_block() {
    return this.u.justPressedKey('n');
  }

  get restart_block() {
    return this.u.justPressedKey('r');
  }

  get moveLeft() {
    const stick = this.u.leftStick();
    const d = this.u.dpad();
    return (
      this.u.isKey('ArrowLeft') ||
      (this.u.isMouseDown() && mouseX < width/2) ||
      stick.x < -0.3 ||
      d.left
    );
  }

  get moveRight() {
    const stick = this.u.leftStick();
    const d = this.u.dpad();
    return (
      this.u.isKey('ArrowRight') ||
      (this.u.isMouseDown() && mouseX > width/2) ||
      stick.x > 0.3 ||
      d.right
    );
  }
}
