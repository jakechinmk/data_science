# Description
This project is to find the top 5 obvious patterns from the [US Traffic 2015 Dataset](https://www.kaggle.com/jboysen/us-traffic-2015)

# Guide
1. Activate your anaconda prompt or open up your terminal
2. Change directory to this file
3. Create conda environment with following command (change <env> accordingly)
```bash
conda create --name <env> --file requirements.txt
```
4. Create a directory data and copy the downloaded data into the directory. Your folder structure will be
```
traffic_flow_us_2015
|- notebook
|-- 01-EDA-01 Exploration.iypnb
|-- 01-EDA-01 Exploration.md
|- data
|-- dot_traffic_2015.txt.gz
|-- dot_traffic_stations_2015.txt.gz
|- requirements.txt
```
5. Activate your environment and run your jupyter notebook
```bash
conda activate <env>
jupyter notebook
```

6. Run 01-EDA-01 Exploration.ipynb to view the complete analysis.
