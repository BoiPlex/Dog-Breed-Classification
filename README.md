# Dog-Breed-Classification
ECS 170

## Setup

```bash
conda create -n <your_env_name> python=3.9
pip install -r requirements.txt
```

> [!IMPORTANT]  
> To run the server successfully, your folder structure should at least look like this:
> ```
> project-170/
> ├── data/
> │   ├── images/
> │   ├── test_data.mat
> │   ├── train_data.mat
> │   ├── Annotation/
> │   └── lists/ 
> │       ├── file_list.mat
> │       ├── tesy_list.mat
> │       └── train_list.mat  
> ├── data_loader.py               
> ├── model.py                     
> ├── train.py                    
> ├── evaluate.py                  
> └── main.py                      
