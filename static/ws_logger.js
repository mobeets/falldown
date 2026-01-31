class WSLogger {
  constructor(url, reconnectInterval = 2000, maxBufferSize = 5000, batchInterval = 500) {
    this.url = url;
    this.reconnectInterval = reconnectInterval;
    this.maxBufferSize = maxBufferSize;
    this.socket = null;

    this.buffer = [];       // immediate events when WS down
    this.batchQueue = [];   // batched events
    this.batchInterval = batchInterval;
    this.log_filename;

    this.connect();
    this._startBatching();
  }

  _generateFilename() {
    const params = new URLSearchParams(window.location.search);
    const subj = params.get("subject_id") || "unknown";
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const rand = Math.random().toString(36).slice(2, 6);
    return `${subj}-${timestamp}-${rand}.jsonl`;
  }

  connect() {
    try {
      this.socket = new WebSocket(this.url);
    } catch (err) {
      console.error("WebSocket failed to construct:", err);
      setTimeout(() => this.connect(), this.reconnectInterval);
      return;
    }

    this.socket.onopen = () => {
      console.log("WebSocket connected");

      // Tell server what file to write to
      this.log_filename = this._generateFilename();
      this.socket.send(JSON.stringify({
        type: "set_filename",
        filename: this.log_filename
      }));
      console.log("Sent filename:", this.log_filename);

      this.flushBuffer();
    };

    this.socket.onclose = () => {
      console.warn(`WebSocket disconnected, reconnecting in ${this.reconnectInterval} ms`);
      setTimeout(() => this.connect(), this.reconnectInterval);
    };

    this.socket.onerror = (err) => {
      console.error("WebSocket error:", err);
      try { this.socket.close(); } catch (_) {}
    };
  }

  log(type, payload = {}, batchable = false, callback) {
    // if batchable is false and callback is a function, we call the callback right after sending
    const evt = {
      type,
      t_wall: Date.now(),
      t_sketch: performance.now(),
      payload: payload
    };
    console.log(evt);

    if (batchable) {
      this._enqueueBatch(evt);
    } else {
      try {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
          this.socket.send(JSON.stringify(evt));
          if (typeof callback === "function") callback();
        } else {
          this._enqueue(evt);
        }
      } catch (err) {
        console.warn("Failed to send WebSocket message, buffering instead", err);
        this._enqueue(evt);
      }
    }
  }

  // Internal enqueue for immediate/buffered events
  _enqueue(evt) {
    this.buffer.push(evt);
    if (this.buffer.length > this.maxBufferSize) {
      console.error('Dropping logs due to maxBufferSize!');
      this.buffer.shift();
    }
  }

  // Internal enqueue for batchable events
  _enqueueBatch(evt) {
    this.batchQueue.push(evt);
    if (this.batchQueue.length > this.maxBufferSize) {
      // drop oldest batchable events
      console.error('Dropping logs due to maxBufferSize!');
      this.batchQueue.shift();
    }
  }

  // flush immediate buffer
  flushBuffer() {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) return;

    while (this.buffer.length > 0) {
      const evt = this.buffer.shift();
      try {
        this.socket.send(JSON.stringify(evt));
      } catch (err) {
        console.warn("Failed to flush buffered message, re-queuing", err);
        this._enqueue(evt);
        break;
      }
    }
  }

  flushAll() {
    // Flush immediate events
    this.flushBuffer();

    // Flush batched events if socket is open
    if (this.batchQueue.length > 0 && this.socket && this.socket.readyState === WebSocket.OPEN) {
      const batchEvent = {
        type: "batch",
        t_wall: Date.now(),
        t_sketch: performance.now(),
        events: this.batchQueue
      };
      try {
        this.socket.send(JSON.stringify(batchEvent));
        this.batchQueue = [];
      } catch (err) {
        console.warn("Failed to flush all batch events", err);
        // keep batchQueue so events are not lost
      }
    }
  }

  saveJson(finalObject) {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      console.warn("Cannot save JSON - WebSocket not connected");
      return;
    }

    try {
      this.socket.send(JSON.stringify({
        type: "save_json",
        contents: JSON.stringify(finalObject)
      }));
      console.log("Saved JSON file");
    } catch (err) {
      console.error("Failed to save JSON", err);
    }
  }

  // batching logic
  _startBatching() {
    setInterval(() => {
      if (!this.socket || this.socket.readyState !== WebSocket.OPEN) return;
      if (this.batchQueue.length === 0) return;

      const batchEvent = {
        type: "batch",
        t_wall: Date.now(),
        t_sketch: performance.now(),
        events: this.batchQueue
      };

      try {
        this.socket.send(JSON.stringify(batchEvent));
        this.batchQueue = [];
      } catch (err) {
        console.warn("Failed to send batch, keeping events", err);
      }
    }, this.batchInterval);
  }
}

// global instance
const wsUrl = `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}/ws`;
window.wsLogger = new WSLogger(wsUrl);

// Call at the end of the sketch, or on page unload
window.addEventListener("beforeunload", () => {
  if (window.wsLogger) window.wsLogger.flushAll();
});
