
### Get dependency from .ipynb
```
jupyter nbconvert --output-dir="./reqs" --to script *.ipynb
cd reqs
pipreqs --print
```
