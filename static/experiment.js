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
    participantId: 'unknown',
    assignmentId: 'unknown',
    projectId: 'unknown',
    params_name: 'cloudresearch_params',
    experiment: 'short_trials_experiment-7-10'
  };

  // Merge defaults with URL params
  const urlParams = getUrlParams();
  const finalParams = { ...defaults, ...urlParams };

  // Safe destructuring with fallback defaults
  const {
    participantId = 'unknown',
    assignmentId = 'unknown',
    projectId = 'unknown',
    experiment = 'default_experiment'
  } = finalParams;

  // Create subject ID
  let subject_id = participantId;
  let isCloudStudy = 0;
  if (assignmentId !== 'unknown') {
    subject_id += '-' + assignmentId;
    isCloudStudy += 1;
  }
  if (projectId !== 'unknown') {
    subject_id += '-' + projectId;
    isCloudStudy += 1;
  }

  // If subject_id has assignmentId and projectId, this must be a CloudResearch study
  let params_name = finalParams.params_name;
  if (params_name === undefined) {
    // n.b. we will get here if params_name was not passed in by url
    if (isCloudStudy === 2) {
      params_name = 'cloudresearch_params';
    } else {
      params_name = 'default_params';
    }
  }

  // Build path to params and experiment files
  const params_path = `configs/${params_name}.json`;
  const experiment_path = `configs/${experiment}.json`;
  
  // In preload(), loadJSON() returns the parsed JSON synchronously
  const params = loadJSON(params_path);
  const block_configs = loadJSON(experiment_path);

  return {subject_id, params_path, experiment_path, params, block_configs};
}

class Experiment {
  constructor({subject_id, params_path, experiment_path, params, block_configs}) {
    this.subject_id = subject_id;
    this.params_path = params_path;
    this.params = params;
    this.experiment_path = experiment_path;
    this.block_configs = block_configs;
    this.block_index = -1;
		this.block_count = -1;
    this.blocks = [];
  }

  next_block(restartGame, goBack) {
    if (!restartGame && !goBack) {
			if (this.blocks.length > 0) {
				// log end of block
				this.blocks[this.blocks.length-1].log(false);
			}
			if (this.no_more_blocks()) {
				this.log(false);
        // log experiment
        wsLogger.saveJson(this);
				return;
			};
			this.block_index++;
		} else if (goBack) {
      // go to previous block
			if (this.block_index >= 1) this.block_index--;
		}
    // log experiment
    wsLogger.saveJson(this);

		this.block_count++;
		let block = new TrialBlock(this.block_index, this.block_count, this.block_configs[this.block_index]);
    block.log(true);
		this.blocks.push(block);
		return block;
  }

  log(isNew = true) {
		let msg = "start of Experiment";
		if (!isNew) msg = "end of Experiment";
		wsLogger.log(msg, this.toJSON());
	}

  toJSON() {
    // outputs all of object's variables as a json object
    return Object.assign({}, this);
  }

  no_more_blocks() {
		return this.block_index+1 >= Object.keys(this.block_configs).length;
	}

	is_complete() {
		return this.no_more_blocks();
	}
}

class TrialBlock {
  constructor(block_index, block_count, block_config) {
    this.block_count = block_count;
		this.block_index = block_index;
    this.block_config = block_config;
    this.trials = [];
    this.trial_index = -1;
    this.last_trial_completed = 0;

    this.start_time;
    this.pause_times = {starts: [], ends: []};
    this.game_states = {time: [], ball_x: [], ball_y: [], ball_vx: [], ball_vy: [], camera_y: [], scroll_speeds: []};
    this.user_inputs = {time: [], input: []};
  }

  log_user_input(input) {
    this.user_inputs.time.push(performance.now());
    this.user_inputs.input.push(input);
  }

  log_states(ball) {
    this.game_states.time.push(performance.now());
    this.game_states.ball_x.push(ball.x);
    this.game_states.ball_y.push(ball.y);
    this.game_states.ball_vx.push(ball.vx);
    this.game_states.ball_vy.push(ball.vy);
    this.game_states.camera_y.push(cameraY);
    this.game_states.scroll_speeds.push(scrollSpeed);
  }

  log_pause_start() {
    this.pause_times.starts.push(performance.now());
  }

  log_pause_end() {
    let now = performance.now();
    if (this.start_time === undefined) {
      this.start_time = now;
    }
    this.pause_times.ends.push(now);
  }

  get_elapsed_time() {
    let t = performance.now() - this.start_time;
    // now subtract all pause durations
    for (var i = this.pause_times.starts.length - 1; i >= 0; i--) {
      t -= (this.pause_times.ends[i+1] - this.pause_times.starts[i]);
    }
    return t;
  }

  is_complete() {
		return this.last_trial_completed >= this.block_config.levels.length;
	}
  
  log(isNew = true) {
		let msg = "start of TrialBlock";
		if (!isNew) msg = "end of TrialBlock";
		wsLogger.log(msg, this.toJSON());
	}

  next_trial() {
		if (this.trial_index + 1 >= this.block_config.levels.length) {
			return;
		}
		
    // todo: log end of previous trial
		this.trial_index++;
		let hole_locations = this.block_config.levels[this.trial_index];

		let trial = new Trial(this.trial_index, this.block_index, hole_locations);
		trial.log(true);
		this.trials.push(trial);
		return trial;
	}

  toJSON() {
    // outputs all of object's variables as a json object
    return Object.assign({}, this);
  }
}

class Trial {
  constructor(index, block_index, hole_locations) {
		this.index = index;
		this.block_index = block_index;
		this.hole_locations = hole_locations;
		this.events = [];
	}

  log(isNew = true) {
		let msg = "start of Trial";
		if (!isNew) msg = "end of Trial";
		wsLogger.log(msg, this.toJSON());
	}

  logEvent(event, callback) {
		event.trial_index = this.index;
		event.hole_locations = this.hole_locations;
		event.block_index = this.block_index;
		event.time = performance.now();
		wsLogger.log("Trial event", event, false, callback);
	}

  trigger(event, callback) {
		if (typeof event === "string") {
			event = {name: event};
		}
		this.logEvent(event, callback);
		this.events.push(event);
	}

  toJSON() {
    // outputs all of object's variables as a json object
    return Object.assign({}, this);
  }
}