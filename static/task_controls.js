class TaskControls {
  constructor(unified) {
    this.u = unified;
  }

  get pause() {
    return this.u.justPressedKey('p') || this.u.justPressedButton("START");
  }

  get next_block() {
    return this.u.justPressedKey('n');
  }

  get restart_block() {
    return this.u.justPressedKey('r');
  }

  get save() {
    return this.u.justPressedKey('s');
  }

  get fired() {
    if (this.fireA) return 1;
    if (this.fireB) return 2;
    if (this.fireC) return 3;
    return -1;
  }

  get fireA() {
    return this.u.justPressedKey('1') || this.u.justPressedButton("WEST");
  }

  get fireB() {
    return this.u.justPressedKey('2') || this.u.justPressedButton("SOUTH");
  }

  get fireC() {
    return this.u.justPressedKey('3') || this.u.justPressedButton("EAST");
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
