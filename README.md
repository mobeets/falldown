## Task instructions

In this experiment we are interested in what happens in the brain when we make decisions that involve planning ahead. In this game you are going to control a ball falling down through a maze. Your goal is to find the fastest path through the maze by choosing when to go left or right. Sometimes finding the fastest path will require looking ahead to the next parts of the maze.

Each maze will take around five minutes to complete, and we'll aim to get through about six mazes in total. Remember to try to move as quickly as you can through each maze.

## Requirements

- [uv](https://docs.astral.sh/uv/#installation)

## Starting the task

To start the server, run the following in a terminal:

`uv run python app/server.py`

To start the task, open a Chrome browser and navigate to `http://0.0.0.0:8000?participantId=SUBJECT_ID`.

This will save all trial data, mouse clicks, and key presses locally to files named `logs/SUBJECT_ID-....jsonl` and `logs/SUBJECT_ID-....json`.

## Experiment details

- Default params can be found in `app/configs/default_params.json`, and overrided by appending `&params_name=example` to the url (this will load the config file `app/configs/example.json`)
- Default block order can be found in `app/configs/default_experiment.json`, and overrided by appending `&experiment=experiment` to the url (this will load the experiment file `app/configs/experiment.json`)

## Controls

On the experimenter's side:
- `p` or SPACEBAR to pause
- `s` to manually save json

On the patient's side (with USB controller connected):
- Left analog stick controls position (left and right only)
- START button or SPACEBAR to pause

For patients with a keyboard:
- Left and right arrow keys control position
- SPACEBAR to pause

All task controls can be found in `app/static/task_controls.js`.

## Debugging

It is recommended to keep the Web Inspector in Chrome open in a separate window so you can make sure the WebSocket remains connected.
