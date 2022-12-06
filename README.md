### Insert here the executive summary once it is done.

Slides: https://docs.google.com/presentation/d/1LXup8CaDlZFC_feaj2BzeHCIj5Nevm75pVJNJSUXTYo/edit?skip_itp2_check=true#slide=id.p
Executive Summary: https://docs.google.com/document/d/1OH3KMq_FwH_pfcZN56qx6OIA_jXwMwQan2YcQj9x-A0/edit

### Get dependency from .ipynb
```
jupyter nbconvert --output-dir="./reqs" --to script *.ipynb
cd reqs
pipreqs --print
```
