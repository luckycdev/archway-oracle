# Archway Oracle
Archway Oracle is a traffic detection system for St. Louis that combines computer vision and machine learning to monitor and predict traffic patterns. It uses a YOLO model to analyze live traffic camera feeds and detect vehicles in real time. The system then feeds this data into a HistGradientBoostingRegressor model to forecast traffic flow across the city. By integrating these two approaches, Archway Oracle provides accurate, real-time insights to help manage traffic congestion and improve transportation planning.

## Roadmap / to-do list
- Make sure live feed is live frames
- Make sure all stats are good
- Make sure selecting camera from dropdown works
- Draw roads on map
- Make sure all previous archway features work
- Make sure ui buttons are in correct order for camera stuff
- Get rid of old google maps stuff
- Look into what console is spamming
- Maybe remove AI Performance section
- Check why map dark/light mode isnt instant change
- Maybe limit the amount of nearby cams
- Use .env and Docker

## Configuration
- Runtime and tuning settings now live in `.env`.
- Typed env parsing and defaults live in `src/config.py`.
- Core modules (`src/app.py`, `src/data_processing.py`, `src/model.py`, `src/engine.py`, `src/camera_get_cams.py`, `src/camera_workers.py`) consume values from `src/config.py`.