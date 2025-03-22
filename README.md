Alexander Shadrov 972301

How to use it:

clone repo via
git clone https://github.com/DaemoNandAndroiD/TSUML-Lab1.git -b master
cd TSUML-Lab1
cd src
cd spaceship
poetry install

After that for train:
poetry run python model.py(or catboost_model.py) train --dataset=./data/train.csv

model wiil be in ./model/modelname.pkl

And for predict:
poetry run python model.py(or catboost_model.py) predict --dataset=./data/test.csv

result file will be in result.csv
all of the logs you can see in log_file.log

i cant select my submission on kaggle due to deadline but i have accuracy

0.794 for catboost & optuna
0.79074 for logistic regression model

![image](https://github.com/user-attachments/assets/8f3b7952-d3a7-4035-80da-43ba3bf1a6c9)

also i have deployed catboost_model on ClearML 
here is screenshots

![image](https://github.com/user-attachments/assets/e0463fa1-af22-4a02-a810-8ed2d47b7e5c)
![image](https://github.com/user-attachments/assets/882f54fe-1382-4c08-8b7f-fa2a090be106)
![image](https://github.com/user-attachments/assets/6235de33-b60a-4f9e-9ab7-2e72a7da45f2)
![image](https://github.com/user-attachments/assets/fd9dad31-2710-4ada-8149-ea57caa20e94)





