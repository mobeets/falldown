class GamepadWrapper {
  constructor(index = 0) {
    this.index = index;

    // Standard Gamepad button mapping (based on Gamepad API standard)
    this.buttonMap = {
      "SOUTH": 0,
      "EAST": 1,
      "WEST": 2,
      "NORTH": 3,
      "LB": 4,
      "RB": 5,
      "LT": 6,
      "RT": 7,
      "BACK": 8,
      "START": 9,
      "LS": 10,
      "RS": 11,
      "DPAD_UP": 12,
      "DPAD_DOWN": 13,
      "DPAD_LEFT": 14,
      "DPAD_RIGHT": 15
    };
  }

  get gp() {
    return navigator.getGamepads()?.[this.index];
  }

  update() {
    // nothing to store; gp remains a getter so it’s always fresh
  }

  // === BUTTON HELPERS ===

  pressed(name) {
    const id = this.buttonMap[name];
    return this.gp?.buttons?.[id]?.pressed || false;
  }

  value(name) {
    const id = this.buttonMap[name];
    return this.gp?.buttons?.[id]?.value || 0;
  }

  // === STICK HELPERS ===

  leftStick() {
    return {
      x: this.gp?.axes?.[0] || 0,
      y: this.gp?.axes?.[1] || 0
    };
  }

  rightStick() {
    return {
      x: this.gp?.axes?.[2] || 0,
      y: this.gp?.axes?.[3] || 0
    };
  }

  // === DPAD HELPERS ===
  get dpad() {
    return {
      up: this.pressed("DPAD_UP"),
      down: this.pressed("DPAD_DOWN"),
      left: this.pressed("DPAD_LEFT"),
      right: this.pressed("DPAD_RIGHT")
    };
  }

  // === CONNECTION STATE ===

  connected() {
    return !!this.gp;
  }

  id() {
    return this.gp?.id || null;
  }
}
