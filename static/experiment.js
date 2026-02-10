function getUrlParams() {
	// parses index.html?key1=val1&key2=val2 to {key1: val1, ...}
  const params = {};
  for (const [key, val] of new URLSearchParams(window.location.search)) {
    params[key] = val;
  }
  return params;
}

function loadConfig() {
  // loads config named in url params
  // call inside preload

  const defaults = {
    subject: 'unknown',
    params_name: 'default_params',
    experiment: 'default_trials'
  };

  // Merge defaults with URL params
  const urlParams = getUrlParams();
  const finalParams = { ...defaults, ...urlParams };

  // Safe destructuring with fallback defaults
  const {
    subject = 'unknown',
    params_name ='default_params',
    experiment = 'default_trials'
  } = finalParams;

  // Build path to params and experiment files
  const params_path = `configs/${params_name}.json`;
  const experiment_path = `configs/${experiment}.json`;
  
  // In preload(), loadJSON() returns the parsed JSON synchronously
  const params = loadJSON(params_path);
  const block_configs = loadJSON(experiment_path);

  return {subject, params_path, experiment_path, params, block_configs};
}

class Experiment {
  constructor({subject_id, params_path, experiment_path, params, block_configs}) {
    this.subject_id = subject_id;
    this.params_path = params_path;
    this.params = params;
    this.experiment_path = experiment_path;
    this.block_configs = block_configs;
    this.block_index = -1;
    this.blocks = [];
  }

  new_block() {
    this.block_index++;
    let block = new TrialBlock(this.block_index);
    this.blocks.push(block);
    return block;
  }

  toJSON() {
    // outputs all of object's variables as a json object
    return Object.assign({}, this);
  }
}

class TrialBlock {
  constructor(index) {
    this.block_index = index;
    this.trials = [];
    this.trial_index = -1;
    this.startTime = millis();
  }
  
  add_trial(level) {
    this.trial_index++;

    let trial = level.toJSON();
    trial.trial_index = this.trial_index;
    trial.block_index = this.block_index;
    trial.timePassedThru = millis() - this.startTime;
    trial.cameraMode = cameraMode;
    trial.ballX = ball.x;
    trial.ballY = ball.y;
    trial.cameraY = cameraY;
    this.trials.push(trial);
  }

  toJSON() {
    // outputs all of object's variables as a json object
    return Object.assign({}, this);
  }
}
