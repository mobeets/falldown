class UnifiedControls {
  constructor(logger = null) {
    this.logger = logger;

    // Gamepad
    this.pad = new GamepadWrapper(0);

    // Keyboard state
    this.keys = {};
    this.prevKeys = {};
    this.justPressedKeys = {};
    this.justReleasedKeys = {};

    // Mouse state
    this.mouseDown = false;
    this.prevMouseDown = false;

    // Gamepad button edges
    this.prevButtons = {};
  }

  keyEventObj(event) { return {code: event.code, key: event.key, keyCode: event.keyCode}; }
  mouseEventObj(event) { return {x: event.x, y: event.y}; }

  // ----- Keyboard -----
  keyPressed(event) {
    this.keys[event.key] = true;
    this.logEvent(`key_pressed`, this.keyEventObj(event));
  }
  keyReleased(event) {
    this.keys[event.key] = false;
    this.logEvent(`key_released`, this.keyEventObj(event));
  }

  // ----- Mouse -----
  mousePressed(event) {
    this.mouseDown = true;
    this.logEvent(`mouse_pressed`, this.mouseEventObj(event));
  }
  mouseReleased(event) {
    this.mouseDown = false;
    this.logEvent(`mouse_released`, this.mouseEventObj(event));
  }

  // ----- Logging helper -----
  logEvent(eventName, event) {
    if (this.logger) {
      this.logger.log(eventName, event);
    }
  }

  // ----- Update per frame -----
  update() {
    this.pad.update();

    // --- keyboard edges
    this.justPressedKeys = {};
    this.justReleasedKeys = {};
    for (let k in this.keys) {
      const now = this.keys[k];
      const prev = this.prevKeys[k] || false;
      this.justPressedKeys[k] = now && !prev;
      this.justReleasedKeys[k] = !now && prev;
    }
    this.prevKeys = { ...this.keys };

    // --- mouse edges
    this.justPressedMouse = this.mouseDown && !this.prevMouseDown;
    this.justReleasedMouse = !this.mouseDown && this.prevMouseDown;
    this.prevMouseDown = this.mouseDown;

    // --- gamepad edges
    const curButtons = {};
    for (const name of Object.keys(this.pad.buttonMap)) {
      curButtons[name] = this.pad.pressed(name);
    }
    this.justPressedButtons = {};
    this.justReleasedButtons = {};
    for (const name in curButtons) {
      const now = curButtons[name];
      const prev = this.prevButtons[name] || false;
      this.justPressedButtons[name] = now && !prev;
      this.justReleasedButtons[name] = !now && prev;

      // log edges
      if (this.justPressedButtons[name]) this.logEvent(`gp_${name}_pressed`);
      if (this.justReleasedButtons[name]) this.logEvent(`gp_${name}_released`);
    }
    this.prevButtons = curButtons;
  }

  // ----- Helpers -----
  isKey(k) { return !!this.keys[k]; }
  justPressedKey(k) { return !!this.justPressedKeys[k]; }
  justReleasedKey(k) { return !!this.justReleasedKeys[k]; }

  isMouseDown() { return this.mouseDown; }
  justPressedMouse() { return this.justPressedMouse; }
  justReleasedMouse() { return this.justReleasedMouse; }

  pressedButton(name) { return this.pad.pressed(name); }
  justPressedButton(name) { return !!this.justPressedButtons[name]; }
  justReleasedButton(name) { return !!this.justReleasedButtons[name]; }

  leftStick() { return this.pad.leftStick(); }
  dpad() { return this.pad.dpad; }
}
