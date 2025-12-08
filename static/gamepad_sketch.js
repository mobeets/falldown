let pad;

function setup() {
  createCanvas(700, 500);
  pad = new GamepadWrapper(0); // use gamepad index 0
}

function draw() {
  background(30);

  pad.update();

  fill(255);
  textSize(16);

  if (!pad.connected()) {
    text("No controller detected", 20, 30);
    return;
  }

  // Show controller name
  text("Controller: " + pad.id(), 20, 30);

  let y = 60;

  // === DISPLAY ALL BUTTONS ===
  text("Buttons:", 20, y);
  y += 20;

  for (const [name, index] of Object.entries(pad.buttonMap)) {
    const pressed = pad.pressed(name);
    const val = pad.value(name).toFixed(2);

    fill(pressed ? "lime" : "white");
    text(`${name} (${index}): ${pressed ? "PRESSED" : val}`, 40, y);

    y += 20;
  }

  y += 10;

  // === DISPLAY STICK POSITIONS ===
  let ls = pad.leftStick();
  let rs = pad.rightStick();

  fill(255);
  text("Left Stick:", 20, y);
  y += 20;
  text(`x: ${ls.x.toFixed(2)}   y: ${ls.y.toFixed(2)}`, 40, y);
  y += 30;

  text("Right Stick:", 20, y);
  y += 20;
  text(`x: ${rs.x.toFixed(2)}   y: ${rs.y.toFixed(2)}`, 40, y);
  y += 40;

  // === DISPLAY DPAD ===
  fill(255);
  text("D-Pad:", 20, y);
  y += 20;

  text(`UP:    ${pad.dpad.up}`,    40, y); y += 20;
  text(`DOWN:  ${pad.dpad.down}`,  40, y); y += 20;
  text(`LEFT:  ${pad.dpad.left}`,  40, y); y += 20;
  text(`RIGHT: ${pad.dpad.right}`, 40, y);
}


// function setup() {
//   createCanvas(400, 400);
// }

// function draw() {
//   background(220);

//   let gamepads = navigator.getGamepads();
//   let gp = gamepads[0];
//   if (!gp) {
//     text("Connect a controller", 20, 20);
//     return;
//   }

//   // Sticks
//   let x = gp.axes[0];
//   let y = gp.axes[1];

//   // Button A / cross / etc
//   let a = gp.buttons[0].pressed;
//   let b = gp.buttons[1].pressed;
//   let c = gp.buttons[2].pressed;
//   let d = gp.buttons[3].pressed;

//   text("Stick: " + x.toFixed(2) + ", " + y.toFixed(2), 20, 50);
//   text("Button a: " + a, 20, 80);
//   text("Button b: " + b, 20, 110);
//   text("Button c: " + c, 20, 140);
//   text("Button d: " + d, 20, 170);

//   // Draw a circle controlled by the stick
//   circle(width/2 + x * 100, height/2 + y * 100, 30);
// }
