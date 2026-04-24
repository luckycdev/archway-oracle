# Archway Oracle
Archway Oracle is a traffic detection system for St. Louis that combines computer vision and machine learning to monitor and predict traffic patterns. It uses a YOLO model to analyze live traffic camera feeds and detect vehicles in real time. The system then feeds this data into a HistGradientBoostingRegressor model to forecast traffic flow across the city. By integrating these two approaches, Archway Oracle provides accurate, real-time insights to help manage traffic congestion and improve transportation planning.

This project is a merge and improvement of [Eagle Eye Traffic Detector](https://github.com/luckycdev/traffic-detector) and [Archway AI Traffic Predictor](https://github.com/DanLDevs/traffic-predictor)

## Roadmap / to-do list
- Draw roads on map
- Make sure all previous archway features work
- Make sure ui buttons are in correct order for camera stuff
- Maybe remove AI Performance section
- Check why map dark/light mode isnt instant change
- Maybe limit the amount of nearby cams
- Use .env and Docker
- limit cams to stl
- use aadt thing


- note: may have to install fmmpeg and torch