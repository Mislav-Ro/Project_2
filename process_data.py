#!/usr/bin/env python
# coding: utf-8

# # ETL Pipeline Preparation
import pandas as pd
import sqlalchemy as db

# load messages dataset
messages = pd.read_csv("raw_data/messages.csv")
categories = pd.read_csv("raw_data/categories.csv")

# merge datasets
df = messages.merge(right=categories, how="inner", on="id")

# create a dataframe of the 36 individual category columns
categoriesX = df.categories.str.split(pat=";", expand=True).copy()
categoriesX.head()

# select the first row of the categories dataframe
row = categoriesX.iloc[0]
category_colnames = row.str.slice(stop=-2)

# rename the columns of `categories`
categoriesX.columns = category_colnames
categoriesX.head()

for column in categoriesX:
    # set each value to be the last character of the string
    categoriesX[column] = categoriesX[column].str.slice(start=-1)
    
    # convert column from string to numeric
    categoriesX[column] = categoriesX[column].astype(int)

# drop the original categories column from `df`
df.drop(columns="categories",inplace=True)

# concatenate the original dataframe with the new `categories` dataframe
df = df.merge(right=categoriesX,how="outer", left_index=True, right_index=True)

# check number of duplicates
# df.duplicated().sum()

# drop duplicates
df.drop_duplicates(inplace=True)

# check number of duplicates
# df.duplicated().sum()

df.loc[df["related"] ==2, "related"] = 1

engine = db.create_engine('sqlite:///MainTable.db')
df.to_sql('info', engine, index=False)