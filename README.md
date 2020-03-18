# AlBert2

## Command line instructions

### Git global setup
```
git config --global user.name "haodong"
git config --global user.email "haodong@cloudwalk.cn"
```

### Create a new repository
```
git clone https://gitlab-research.cloudwalk.work/NLP/AP_10_ALBERT2.git
cd AP_10_ALBERT2
touch README.md
git add README.md
git commit -m "add README"
git push -u origin master
```

### Existing folder
```
cd existing_folder
git init
git remote add origin https://gitlab-research.cloudwalk.work/NLP/AP_10_ALBERT2.git
git add .
git commit -m "Initial commit"
git push -u origin master
```

### Existing Git repository
```
cd existing_repo
git remote rename origin old-origin
git remote add origin https://gitlab-research.cloudwalk.work/NLP/AP_10_ALBERT2.git
git push -u origin --all
git push -u origin --tags
```