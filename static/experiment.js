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
    experiment: 'default_experiment'
  };

  // Merge defaults with URL params
  const urlParams = getUrlParams();
  const finalParams = { ...defaults, ...urlParams };

  // Safe destructuring with fallback defaults
  const {
    subject = 'unknown',
    params_name ='default_params',
    experiment = 'default_experiment'
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
				// log end of experiment
				wsLogger.saveJson(this);
				this.log(false);
				return;
			};
			this.block_index++;
		} else if (goBack) {
			if (this.block_index >= 1) this.block_index--;
		}

		this.block_count++;
		let block = new TrialBlock(this.block_index, this.block_count, this.block_configs[this.block_index]);
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
    // this.startTime = millis();
  }

  is_complete() {
		return this.trials.length >= this.block_config.levels.length;
	}
  
  log(isNew = true) {
		let msg = "start of TrialBlock";
		if (!isNew) msg = "end of TrialBlock";
		wsLogger.log(msg, this.toJSON());
	}

  next_trial() {
		if (this.is_complete()) {
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