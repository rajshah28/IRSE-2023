import pandas as pd

df = pd.read_excel('data.xlsx')

for index,row in df.iterrows():
    code = str(df.at[index,'code_snippet'])
    comment = str(df.at[index,'comments'])
    df.at[index,'Query'] = "State whether the comment block given for the code snippet is useful or not. Code: \n"+code+"\nComment: "+comment+"\nOnly return Useful or not useful."

df.to_excel('data.xlsx', index=False)